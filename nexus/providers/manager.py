"""
Provider manager for NEXUS.

Manages cloud provider integrations, handles provider selection, health monitoring,
and job submission across different cloud platforms (Kaggle, Colab, Hugging Face).
"""

import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from nexus.core.config import Config
from nexus.core.exceptions import ProviderError, ProviderUnavailable
from nexus.core.logger import get_logger
from nexus.providers.base import (
    ComputeProvider,
    JobSpec,
    JobHandle,
    JobStatus,
    AuthResult,
    OutputResult,
    Capabilities,
    Availability,
    QuotaInfo,
)


class ProviderManager:
    """Manages cloud provider integrations and job submission."""
    
    def __init__(self, config: Config):
        self.logger = get_logger(__name__)
        self.config = config
        self._providers: Dict[str, ComputeProvider] = {}
        self._provider_configs: Dict[str, Dict[str, Any]] = {}
        self._health_status: Dict[str, Dict[str, Any]] = {}
        
        # Load provider configurations
        self._load_provider_configs()
        
        # Initialize providers
        self._initialize_providers()
        
        # Start health monitoring thread
        self._health_monitor_running = True
        self._health_monitor_thread = None
    
    def _load_provider_configs(self):
        """Load provider configurations from config."""
        providers_config = self.config.get("providers", {})
        
        for provider_name, provider_config in providers_config.items():
            if provider_config.get("enabled", False):
                self._provider_configs[provider_name] = {
                    "enabled": True,
                    "priority": provider_config.get("priority", 1),
                    "auth_env_vars": provider_config.get("auth_env_vars", {}),
                    "timeout_seconds": provider_config.get("timeout_seconds", 30),
                    "max_retries": provider_config.get("max_retries", 3),
                    "retry_delay": provider_config.get("retry_delay", 1.0),
                }
    
    def _initialize_providers(self):
        """Initialize all configured providers."""
        # Import here to avoid circular imports
        from nexus.providers.kaggle import KaggleProvider
        from nexus.providers.colab import ColabProvider
        from nexus.providers.huggingface import HuggingFaceProvider
        
        provider_classes = {
            "kaggle": KaggleProvider,
            "colab": ColabProvider,
            "huggingface": HuggingFaceProvider,
        }
        
        for provider_name, provider_class in provider_classes.items():
            if provider_name in self._provider_configs:
                config = self._provider_configs[provider_name]
                try:
                    provider = provider_class(provider_name, config)
                    self._providers[provider_name] = provider
                    self.logger.info(f"Initialized provider: {provider_name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize provider {provider_name}: {str(e)}")
    
    def get_provider(self, name: str) -> Optional[ComputeProvider]:
        """Get a provider by name."""
        return self._providers.get(name)
    
    def get_all_providers(self) -> List[ComputeProvider]:
        """Get all available providers."""
        return list(self._providers.values())
    
    def get_provider_names(self) -> List[str]:
        """Get list of provider names."""
        return list(self._providers.keys())
    
    def authenticate(self, provider_name: Optional[str] = None) -> Dict[str, AuthResult]:
        """Authenticate with providers."""
        if provider_name:
            providers = [self._providers.get(provider_name)]
        else:
            providers = self.get_all_providers()
        
        results = {}
        
        for provider in providers:
            if not provider:
                continue
            
            try:
                auth_result = provider.authenticate()
                results[provider.name] = auth_result
                self.logger.info(f"Authenticated with provider: {provider.name}")
            except Exception as e:
                results[provider.name] = AuthResult(
                    provider=provider.name,
                    method=provider.config.get("auth_method", "unknown"),
                    authenticated=False,
                    error=str(e)
                )
                self.logger.error(f"Failed to authenticate with provider {provider.name}: {str(e)}")
        
        return results
    
    def check_provider_health(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """Check health of providers."""
        if provider_name:
            providers = [self._providers.get(provider_name)]
        else:
            providers = self.get_all_providers()
        
        results = {}
        
        for provider in providers:
            if not provider:
                continue
            
            try:
                # Check authentication
                auth_result = provider.authenticate()
                
                # Check availability
                availability = provider.check_availability()
                
                # Check quota if available
                quota = provider.quota()
                
                # Get capabilities
                capabilities = provider.capabilities()
                
                results[provider.name] = {
                    "authenticated": auth_result.authenticated,
                    "available": availability.available,
                    "gpu_available": availability.gpu_available,
                    "queue_time": availability.queue_time_seconds,
                    "current_jobs": availability.current_active_jobs,
                    "quota": {
                        "used": quota.used if quota else 0,
                        "limit": quota.limit if quota else 0,
                        "remaining": quota.remaining if quota else 0,
                        "ok": (quota.used if quota else 0) < (quota.limit if quota else float('inf'))
                    } if quota else None,
                    "capabilities": capabilities.__dict__,
                    "last_check": time.time()
                }
                
            except Exception as e:
                results[provider.name] = {
                    "error": str(e),
                    "last_check": time.time()
                }
        
        # Update health status cache
        self._health_status.update(results)
        
        return results
    
    def select_provider(self, job_spec: JobSpec, preferred_providers: Optional[List[str]] = None) -> ComputeProvider:
        """Select the best provider for a job."""
        # Get providers with authentication
        authenticated_providers = []
        
        for provider_name, provider in self._providers.items():
            try:
                # Quick auth check
                auth_result = provider.authenticate()
                if auth_result.authenticated:
                    authenticated_providers.append((provider_name, provider))
            except Exception:
                continue
        
        if not authenticated_providers:
            raise ProviderError(
                "No authenticated providers available",
                provider="all",
                operation="provider_selection"
            )
        
        # Filter by preferred providers
        if preferred_providers:
            filtered_providers = [
                (name, provider) for name, provider in authenticated_providers
                if name in preferred_providers
            ]
        else:
            filtered_providers = authenticated_providers
        
        if not filtered_providers:
            filtered_providers = authenticated_providers
        
        # Sort by priority (lower number = higher priority)
        sorted_providers = sorted(filtered_providers, key=lambda x: self._provider_configs.get(x[0], {}).get("priority", 1))
        
        # Check each provider for capability
        for provider_name, provider in sorted_providers:
            try:
                # Validate job
                errors = provider.validate_job(job_spec)
                if errors:
                    continue
                
                # Check availability
                availability = provider.check_availability()
                if availability.available and availability.gpu_available:
                    # Check quota
                    quota = provider.quota()
                    if quota is None or quota.used < quota.limit:
                        return provider
                
            except Exception as e:
                self.logger.debug(f"Provider {provider_name} not suitable for job {job_spec.job_id}: {str(e)}")
                continue
        
        # If no provider fits, use the highest priority available
        if sorted_providers:
            provider_name, provider = sorted_providers[0]
            self.logger.warning(f"No perfectly suitable provider for job {job_spec.job_id}, using {provider_name}")
            return provider
        
        raise ProviderError(
            "No available providers for job",
            provider="all",
            operation="provider_selection"
        )
    
    def submit_job(self, job_spec: JobSpec, preferred_providers: Optional[List[str]] = None) -> JobHandle:
        """Submit a job to the best available provider."""
        # Select provider
        provider = self.select_provider(job_spec, preferred_providers)
        
        self.logger.info(f"Submitting job {job_spec.job_id} to provider {provider.name}")
        
        try:
            # Submit job
            handle = provider.submit_job(job_spec)
            
            self.logger.info(f"Job {job_spec.job_id} submitted with handle {handle.handle}")
            
            # Update health status
            self._update_provider_health(provider.name)
            
            return handle
            
        except Exception as e:
            raise ProviderError(
                f"Failed to submit job {job_spec.job_id}: {str(e)}",
                provider=provider.name,
                operation="job_submit"
            )
    
    def get_job_status(self, handle: JobHandle) -> JobStatus:
        """Get job status from provider."""
        provider = self._providers.get(handle.provider)
        
        if not provider:
            raise ProviderError(
                f"Provider {handle.provider} not found",
                provider=handle.provider,
                operation="get_status"
            )
        
        try:
            status = provider.get_status(handle)
            
            # Update health status
            self._update_provider_health(handle.provider)
            
            return status
            
        except Exception as e:
            raise ProviderError(
                f"Failed to get job status {handle.job_id}: {str(e)}",
                provider=handle.provider,
                operation="get_status"
            )
    
    def retrieve_job_output(self, handle: JobHandle, dest_path: str) -> OutputResult:
        """Retrieve job output from provider."""
        provider = self._providers.get(handle.provider)
        
        if not provider:
            raise ProviderError(
                f"Provider {handle.provider} not found",
                provider=handle.provider,
                operation="retrieve_output"
            )
        
        try:
            output_result = provider.retrieve_output(handle, dest_path)
            
            # Update health status
            self._update_provider_health(handle.provider)
            
            return output_result
            
        except Exception as e:
            raise ProviderError(
                f"Failed to retrieve job output {handle.job_id}: {str(e)}",
                provider=handle.provider,
                operation="retrieve_output"
            )
    
    def cancel_job(self, handle: JobHandle):
        """Cancel a job."""
        provider = self._providers.get(handle.provider)
        
        if not provider:
            raise ProviderError(
                f"Provider {handle.provider} not found",
                provider=handle.provider,
                operation="cancel_job"
            )
        
        try:
            provider.cancel_job(handle)
            
            # Update health status
            self._update_provider_health(handle.provider)
            
        except Exception as e:
            raise ProviderError(
                f"Failed to cancel job {handle.job_id}: {str(e)}",
                provider=handle.provider,
                operation="cancel_job"
            )
    
    def _update_provider_health(self, provider_name: str):
        """Update health status for a provider."""
        try:
            health = self.check_provider_health(provider_name)
            if provider_name in health:
                self._health_status[provider_name] = health[provider_name]
        except Exception as e:
            self.logger.debug(f"Failed to update health for provider {provider_name}: {str(e)}")
    
    def get_provider_health(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """Get provider health status."""
        if provider_name:
            return self._health_status.get(provider_name, {})
        else:
            return self._health_status.copy()
    
    def wait_for_job(self, handle: JobHandle, timeout: Optional[float] = None,
                     poll_interval: float = 5.0) -> JobStatus:
        """Wait for job completion."""
        start_time = time.time()
        
        while True:
            try:
                status = self.get_job_status(handle)
                
                # Check if job is completed
                if status.state == "completed":
                    return status
                elif status.state == "failed":
                    raise JobStateError(
                        f"Job {handle.job_id} failed: {status.error}",
                        job_id=handle.job_id
                    )
                
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    raise JobStateError(
                        f"Job {handle.job_id} timed out after {timeout} seconds",
                        job_id=handle.job_id
                    )
                
                # Wait before next check
                time.sleep(poll_interval)
                
            except JobStateError:
                raise
            except Exception as e:
                self.logger.warning(f"Error checking job status: {str(e)}")
                time.sleep(poll_interval)
    
    def stop(self):
        """Stop provider manager and clean up."""
        self._health_monitor_running = False
        
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            self._health_monitor_thread.join(timeout=5.0)