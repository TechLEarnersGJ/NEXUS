# NEXUS Providers Module

from nexus.providers.base import (
    AuthMethod,
    Capabilities,
    Availability,
    QuotaInfo,
    JobSpec,
    JobHandle,
    JobStatus,
    AuthResult,
    OutputResult,
    ComputeProvider,
    ProviderHealthChecker,
)

from nexus.providers.manager import ProviderManager
from nexus.providers.fake import (
    FakeKaggleProvider,
    FakeColabProvider,
    FakeHuggingFaceProvider,
    FakeProvider,
)

__all__ = [
    # Base classes
    "AuthMethod",
    "Capabilities",
    "Availability",
    "QuotaInfo",
    "JobSpec",
    "JobHandle",
    "JobStatus",
    "AuthResult",
    "OutputResult",
    "ComputeProvider",
    "ProviderHealthChecker",
    
    # Manager
    "ProviderManager",
    
    # Fake providers
    "FakeKaggleProvider",
    "FakeColabProvider",
    "FakeHuggingFaceProvider",
    "FakeProvider",
]