"""
Core exception definitions for NEXUS.

Provides a comprehensive exception hierarchy for error handling throughout
the NEXUS platform, with specific error types for different failure scenarios.
"""

from typing import Optional, Dict, Any


class NexusError(Exception):
    """Base exception for all NEXUS errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigError(NexusError):
    """Raised when there's an error with configuration."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, error_code="CONFIG_ERROR")
        self.config_key = config_key


class AuthError(NexusError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, provider: Optional[str] = None):
        super().__init__(message, error_code="AUTH_ERROR")
        self.provider = provider


class ProviderError(NexusError):
    """Raised when there's an error with a cloud provider."""
    
    def __init__(self, message: str, provider: str, operation: Optional[str] = None):
        super().__init__(message, error_code="PROVIDER_ERROR")
        self.provider = provider
        self.operation = operation


class ProviderUnavailable(NexusError):
    """Raised when a provider is unavailable or not supported."""
    
    def __init__(self, message: str, provider: str, reason: Optional[str] = None):
        super().__init__(message, error_code="PROVIDER_UNAVAILABLE")
        self.provider = provider
        self.reason = reason


class QuotaExceeded(NexusError):
    """Raised when a provider quota is exceeded."""
    
    def __init__(self, message: str, provider: str, quota_type: str, 
                 used: float, limit: float):
        super().__init__(message, error_code="QUOTA_EXCEEDED")
        self.provider = provider
        self.quota_type = quota_type
        self.used = used
        self.limit = limit


class JobStateError(NexusError):
    """Raised when there's an error with job state management."""
    
    def __init__(self, message: str, job_id: Optional[str] = None, 
                 current_state: Optional[str] = None):
        super().__init__(message, error_code="JOB_STATE_ERROR")
        self.job_id = job_id
        self.current_state = current_state


class ValidationError(NexusError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None):
        super().__init__(message, error_code="VALIDATION_ERROR")
        self.field = field
        self.value = value