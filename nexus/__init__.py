"""
NEXUS - Autonomous AI Model Development Platform

A powerful, modular autonomous AI model development system that can understand
natural language requests, plan work, discover models and datasets, allocate
cloud compute, train/fine-tune models, evaluate results, improve them and
export the finished model.
"""

__version__ = "0.1.0"

# Core imports
from nexus.core.exceptions import (
    NexusError,
    ConfigError,
    AuthError,
    ProviderError,
    ProviderUnavailable,
    QuotaExceeded,
    JobStateError,
    ValidationError,
)

from nexus.core.config import Config
from nexus.core.logger import get_logger
from nexus.core.state import JobManager
from nexus.core.engine import Engine

# Provider imports
from nexus.providers.manager import ProviderManager
from nexus.providers.base import (
    Capabilities,
    Availability,
    QuotaInfo,
    JobSpec,
    JobHandle,
    JobStatus,
    AuthResult,
    OutputResult,
)

# CLI imports
from nexus.cli import main as cli_main

__all__ = [
    # Core
    "NexusError",
    "ConfigError", 
    "AuthError",
    "ProviderError",
    "ProviderUnavailable",
    "QuotaExceeded",
    "JobStateError",
    "ValidationError",
    "Config",
    "get_logger",
    "JobManager",
    "Engine",
    
    # Providers
    "ProviderManager",
    "Capabilities",
    "Availability",
    "QuotaInfo",
    "JobSpec",
    "JobHandle",
    "JobStatus",
    "AuthResult",
    "OutputResult",
    
    # CLI
    "cli_main",
    
    # Version
    "__version__",
]