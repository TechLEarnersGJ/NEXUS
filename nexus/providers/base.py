"""
Base classes and interfaces for NEXUS cloud providers.

Defines the abstract interface that all cloud providers must implement
and common data structures for provider capabilities, job management, and authentication.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from nexus.core.exceptions import (
    AuthError,
    ProviderError,
    ProviderUnavailable,
    QuotaExceeded,
    JobStateError,
    ValidationError,
)


class AuthMethod(Enum):
    """Authentication methods supported by providers."""
    
    API_KEY = "api_key"
    OAUTH = "oauth"
    TOKEN = "token"
    SERVICE_ACCOUNT = "service_account"
    CREDENTIALS_FILE = "credentials_file"


@dataclass
class Capabilities:
    """Provider capabilities."""
    
    gpu_memory_gb: float = 0.0
    max_runtime_h: float = 0.0
    resume: bool = False
    gpu_count: int = 0
    max_memory_gb: float = 0.0
    supports_colab: bool = False
    supports_huggingface: bool = False
    supports_notebooks: bool = False
    supports_custom_images: bool = False
    supports_interactive: bool = False
    supports_monitoring: bool = False


@dataclass
class Availability:
    """Provider availability information."""
    
    available: bool = False
    gpu_available: bool = False
    quota_available: bool = False
    queue_time_seconds: float = 0.0
    estimated_wait_time_seconds: float = 0.0
    max_concurrent_jobs: int = 0
    current_active_jobs: int = 0
    last_check: float = 0.0  # Unix timestamp


@dataclass
class QuotaInfo:
    """Provider quota information."""
    
    gpu_type: str = ""
    quota_type: str = ""
    used: float = 0.0
    limit: float = 0.0
    remaining: float = 0.0
    refresh_at: Optional[float] = None
    quota_id: Optional[str] = None
    reset_time: Optional[float] = None


@dataclass
class JobSpec:
    """Job specification."""
    
    job_id: str
    name: str
    provider: str
    description: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    priority: int = 1
    timeout_seconds: Optional[int] = None
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class JobHandle:
    """Handle for a submitted job."""
    
    job_id: str
    provider: str
    handle: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass
class JobStatus:
    """Job status information."""
    
    job_id: str
    provider: str
    status: str
    state: str
    progress: float = 0.0
    message: Optional[str] = None
    error: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthResult:
    """Authentication result."""
    
    provider: str
    method: AuthMethod
    authenticated: bool
    token: Optional[str] = None
    expires_at: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputResult:
    """Output retrieval result."""
    
    job_id: str
    provider: str
    output_path: str
    output_files: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    download_url: Optional[str] = None
    expires_at: Optional[float] = None
    error: Optional[str] = None


class ComputeProvider(ABC):
    """Abstract base class for cloud compute providers."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self._auth_result: Optional[AuthResult] = None
        self._availability_cache: Optional[Availability] = None
        self._quota_cache: Optional[QuotaInfo] = None
    
    @abstractmethod
    def capabilities(self) -> Capabilities:
        """Get provider capabilities."""
        pass
    
    @abstractmethod
    def authenticate(self) -> AuthResult:
        """Authenticate with the provider."""
        pass
    
    @abstractmethod
    def check_availability(self) -> Availability:
        """Check provider availability."""
        pass
    
    def quota(self) -> Optional[QuotaInfo]:
        """Get provider quota information."""
        return self._quota_cache
    
    @abstractmethod
    def submit_job(self, job: JobSpec) -> JobHandle:
        """Submit a job to the provider."""
        pass
    
    @abstractmethod
    def get_status(self, handle: JobHandle) -> JobStatus:
        """Get job status."""
        pass
    
    @abstractmethod
    def retrieve_output(self, handle: JobHandle, dest_path: str) -> OutputResult:
        """Retrieve job output."""
        pass
    
    def cancel_job(self, handle: JobHandle):
        """Cancel a job (optional operation)."""
        raise NotImplementedError("Job cancellation not supported by this provider")
    
    def get_job_logs(self, handle: JobHandle) -> Optional[str]:
        """Get job logs (optional operation)."""
        return None
    
    def get_job_events(self, handle: JobHandle) -> Optional[List[Dict[str, Any]]]:
        """Get job events (optional operation)."""
        return None
    
    def validate_job(self, job: JobSpec) -> List[str]:
        """Validate a job specification and return list of errors."""
        errors = []
        
        if not job.job_id:
            errors.append("Job ID cannot be empty")
        
        if not job.name:
            errors.append("Job name cannot be empty")
        
        if not job.provider:
            errors.append("Provider cannot be empty")
        
        if job.provider != self.name:
            errors.append(f"Provider mismatch: expected {self.name}, got {job.provider}")
        
        if not isinstance(job.priority, int) or job.priority < 1 or job.priority > 10:
            errors.append("Priority must be an integer between 1 and 10")
        
        if job.timeout_seconds is not None and job.timeout_seconds <= 0:
            errors.append("Timeout must be positive")
        
        return errors
    
    def _validate_auth_result(self, auth_result: AuthResult):
        """Validate authentication result."""
        if not auth_result.authenticated:
            raise AuthError(
                f"Authentication failed for provider {self.name}: {auth_result.error}",
                provider=self.name
            )
    
    def _validate_quota(self, quota: Optional[QuotaInfo]):
        """Validate quota information."""
        if quota is None:
            return  # Quota not available
        
        if quota.used >= quota.limit:
            raise QuotaExceeded(
                f"Quota exceeded for provider {self.name}",
                provider=self.name,
                quota_type=quota.quota_type,
                used=quota.used,
                limit=quota.limit
            )
    
    def _wait_for_availability(self, timeout: float = 30.0) -> bool:
        """Wait for provider to become available."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            availability = self.check_availability()
            if availability.available and availability.gpu_available:
                return True
            
            time.sleep(1.0)
        
        return False


class ProviderHealthChecker:
    """Utility class for checking provider health."""
    
    @staticmethod
    def check_all_providers(providers: List[ComputeProvider]) -> Dict[str, bool]:
        """Check health of all providers."""
        results = {}
        
        for provider in providers:
            try:
                # Check authentication
                auth_result = provider.authenticate()
                if not auth_result.authenticated:
                    results[provider.name] = False
                    continue
                
                # Check availability
                availability = provider.check_availability()
                results[provider.name] = availability.available and availability.gpu_available
                
            except Exception as e:
                results[provider.name] = False
        
        return results
    
    @staticmethod
    def check_provider_health(provider: ComputeProvider) -> Dict[str, Any]:
        """Check health of a single provider."""
        health_status = {
            "name": provider.name,
            "authenticated": False,
            "available": False,
            "quota_ok": False,
            "capabilities": None,
            "last_check": time.time()
        }
        
        try:
            # Check authentication
            auth_result = provider.authenticate()
            health_status["authenticated"] = auth_result.authenticated
            
            if auth_result.authenticated:
                # Check availability
                availability = provider.check_availability()
                health_status["available"] = availability.available
                health_status["queue_time"] = availability.queue_time_seconds
                health_status["current_jobs"] = availability.current_active_jobs
                
                # Check quota
                quota = provider.quota()
                if quota:
                    health_status["quota_ok"] = quota.used < quota.limit
                    health_status["quota_used"] = quota.used
                    health_status["quota_limit"] = quota.limit
                    health_status["quota_remaining"] = quota.remaining
            
            # Get capabilities
            health_status["capabilities"] = provider.capabilities().__dict__
            
        except Exception as e:
            health_status["error"] = str(e)
        
        return health_status