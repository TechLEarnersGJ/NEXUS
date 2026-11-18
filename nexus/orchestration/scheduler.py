"""
Job scheduler for NEXUS autonomous orchestration.

Schedules training jobs across cloud providers, tracks quota usage,
and manages provider selection based on availability and cost.
"""

import time
from typing import Any, Dict, List, Optional

from nexus.core.exceptions import ProviderError, ProviderUnavailable, QuotaExceeded
from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger
from nexus.providers.manager import ProviderManager


class JobScheduler:
    """Schedules jobs across available cloud providers."""
    
    def __init__(
        self,
        provider_manager: ProviderManager,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.provider_manager = provider_manager
        self.config = config or {}
        
        # Scheduling settings
        self.max_concurrent_jobs = self.config.get("max_concurrent_jobs", 3)
        self.default_timeout_hours = self.config.get("default_timeout_hours", 24)
        self.poll_interval_seconds = self.config.get("poll_interval_seconds", 30)
        
        # Track scheduled jobs
        self._scheduled_jobs: Dict[str, Dict[str, Any]] = {}
        self._job_queue: List[str] = []
    
    def schedule_job(
        self,
        job_spec: Dict[str, Any],
        preferred_providers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Schedule a job on the best available provider."""
        self.nexus_logger.log_event(
            "job_scheduling_start",
            {
                "job_id": job_spec.get("job_id"),
                "provider": job_spec.get("provider"),
                "preferred_providers": preferred_providers,
            }
        )
        
        job_id = job_spec.get("job_id", f"job_{int(time.time())}")
        
        # Check if we have capacity
        active_count = len([
            j for j in self._scheduled_jobs.values()
            if j.get("status") in ("queued", "running")
        ])
        
        if active_count >= self.max_concurrent_jobs:
            # Queue the job
            self._job_queue.append(job_id)
            self._scheduled_jobs[job_id] = {
                "job_spec": job_spec,
                "status": "queued",
                "scheduled_at": time.time(),
                "provider": None,
                "handle": None,
            }
            
            self.nexus_logger.log_event(
                "job_queued",
                {
                    "job_id": job_id,
                    "queue_position": len(self._job_queue),
                }
            )
            
            return {
                "scheduled": True,
                "status": "queued",
                "job_id": job_id,
                "message": f"Job queued (position {len(self._job_queue)})",
            }
        
        # Select provider
        provider = self._select_provider(job_spec, preferred_providers)
        
        if not provider:
            raise ProviderUnavailable(
                "No available providers for job scheduling",
                provider="all",
                reason="All providers unavailable or quota exceeded",
            )
        
        # Submit job
        from nexus.providers.base import JobSpec
        
        js = JobSpec(
            job_id=job_spec.get("job_id", job_id),
            name=job_spec.get("name", job_id),
            provider=provider.name,
            config=job_spec.get("config", {}),
            requirements=job_spec.get("requirements", {}),
            resources=job_spec.get("resources", {}),
            metadata=job_spec.get("metadata", {}),
            priority=job_spec.get("priority", 1),
        )
        
        try:
            handle = provider.submit_job(js)
        except Exception as e:
            raise ProviderError(
                f"Failed to submit job {job_id} to provider {provider.name}: {str(e)}",
                provider=provider.name,
                operation="job_submit",
            )
        
        # Track the job
        self._scheduled_jobs[job_id] = {
            "job_spec": job_spec,
            "status": "submitted",
            "scheduled_at": time.time(),
            "provider": provider.name,
            "handle": handle,
        }
        
        self.nexus_logger.log_event(
            "job_scheduled",
            {
                "job_id": job_id,
                "provider": provider.name,
                "handle": handle.handle,
            }
        )
        
        return {
            "scheduled": True,
            "status": "submitted",
            "job_id": job_id,
            "provider": provider.name,
            "handle": handle.handle,
        }
    
    def _select_provider(
        self,
        job_spec: Dict[str, Any],
        preferred_providers: Optional[List[str]] = None,
    ) -> Any:
        """Select the best provider for a job."""
        # Get provider capabilities and availability
        providers = self.provider_manager.get_all_providers()
        
        # Filter by preferred providers
        if preferred_providers:
            providers = [p for p in providers if p.name in preferred_providers]
        
        if not providers:
            return None
        
        # Sort by priority
        provider_configs = self.provider_manager._provider_configs
        sorted_providers = sorted(
            providers,
            key=lambda p: provider_configs.get(p.name, {}).get("priority", 99),
        )
        
        # Check each provider
        for provider in sorted_providers:
            try:
                # Quick auth check
                auth_result = provider.authenticate()
                if not auth_result.authenticated:
                    continue
                
                # Check availability
                availability = provider.check_availability()
                if not availability.available or not availability.gpu_available:
                    continue
                
                # Check quota
                quota = provider.quota()
                if quota and quota.used >= quota.limit:
                    continue
                
                return provider
                
            except Exception:
                continue
        
        return None
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a scheduled job."""
        if job_id not in self._scheduled_jobs:
            return None
        
        job_info = self._scheduled_jobs[job_id]
        
        # If we have a handle, check with provider
        if job_info.get("handle") and job_info.get("provider"):
            provider = self.provider_manager.get_provider(job_info["provider"])
            if provider:
                try:
                    status = provider.get_status(job_info["handle"])
                    job_info["status"] = status.state
                    job_info["provider_status"] = status.__dict__
                except Exception:
                    pass
        
        return {
            "job_id": job_id,
            "status": job_info["status"],
            "provider": job_info.get("provider"),
            "scheduled_at": job_info.get("scheduled_at"),
            "handle": job_info.get("handle").handle if job_info.get("handle") else None,
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job."""
        if job_id not in self._scheduled_jobs:
            return False
        
        job_info = self._scheduled_jobs[job_id]
        
        # Cancel with provider if submitted
        if job_info.get("handle") and job_info.get("provider"):
            provider = self.provider_manager.get_provider(job_info["provider"])
            if provider:
                try:
                    provider.cancel_job(job_info["handle"])
                except Exception:
                    pass
        
        # Remove from queue if queued
        if job_id in self._job_queue:
            self._job_queue.remove(job_id)
        
        # Update status
        job_info["status"] = "cancelled"
        job_info["cancelled_at"] = time.time()
        
        # Process next queued job
        self._process_queue()
        
        self.nexus_logger.log_event(
            "job_cancelled",
            {
                "job_id": job_id,
            }
        )
        
        return True
    
    def _process_queue(self) -> None:
        """Process queued jobs when capacity becomes available."""
        while self._job_queue:
            active_count = len([
                j for j in self._scheduled_jobs.values()
                if j.get("status") in ("queued", "running", "submitted")
            ])
            
            if active_count >= self.max_concurrent_jobs:
                break
            
            job_id = self._job_queue.pop(0)
            job_info = self._scheduled_jobs[job_id]
            
            # Try to schedule
            try:
                provider = self._select_provider(
                    job_info["job_spec"],
                    job_info["job_spec"].get("metadata", {}).get("preferred_providers"),
                )
                
                if not provider:
                    # Put back in queue
                    self._job_queue.insert(0, job_id)
                    break
                
                from nexus.providers.base import JobSpec
                js = JobSpec(
                    job_id=job_info["job_spec"].get("job_id", job_id),
                    name=job_info["job_spec"].get("name", job_id),
                    provider=provider.name,
                    config=job_info["job_spec"].get("config", {}),
                    requirements=job_info["job_spec"].get("requirements", {}),
                    resources=job_info["job_spec"].get("resources", {}),
                    metadata=job_info["job_spec"].get("metadata", {}),
                    priority=job_info["job_spec"].get("priority", 1),
                )
                
                handle = provider.submit_job(js)
                
                job_info["status"] = "submitted"
                job_info["provider"] = provider.name
                job_info["handle"] = handle
                job_info["started_at"] = time.time()
                
                self.nexus_logger.log_event(
                    "job_dequeued_and_scheduled",
                    {
                        "job_id": job_id,
                        "provider": provider.name,
                    }
                )
                
            except Exception as e:
                self.nexus_logger.log_event(
                    "job_queue_processing_error",
                    {
                        "job_id": job_id,
                        "error": str(e),
                    }
                )
                break
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all scheduled jobs."""
        results = []
        for job_id, job_info in self._scheduled_jobs.items():
            results.append({
                "job_id": job_id,
                "status": job_info["status"],
                "provider": job_info.get("provider"),
                "scheduled_at": job_info.get("scheduled_at"),
            })
        return results
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        active = len([
            j for j in self._scheduled_jobs.values()
            if j.get("status") in ("queued", "running", "submitted")
        ])
        completed = len([
            j for j in self._scheduled_jobs.values()
            if j.get("status") in ("completed", "failed", "cancelled")
        ])
        
        return {
            "active_jobs": active,
            "queued_jobs": len(self._job_queue),
            "completed_jobs": completed,
            "max_concurrent": self.max_concurrent_jobs,
            "available_slots": max(0, self.max_concurrent_jobs - active),
        }