"""
Configuration management for NEXUS.

Handles loading, parsing, and validating configuration from multiple sources
with precedence: CLI > Environment Variables > Configuration Files > Defaults.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from nexus.core.exceptions import ConfigError


class LogLevel(Enum):
    """Log levels supported by NEXUS."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Log formats supported by NEXUS."""
    
    TEXT = "text"
    JSON = "json"


@dataclass
class ProviderConfig:
    """Configuration for a cloud provider."""
    
    name: str
    enabled: bool = True
    priority: int = 1
    auth_env_vars: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class TrainingConfig:
    """Configuration for training operations."""
    
    method: str = "qlora"
    batch_size: int = 2
    learning_rate: float = 0.0001
    epochs: int = 3
    max_sequence_length: int = 2048
    gradient_checkpointing: bool = True
    mixed_precision: str = "bfloat16"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_double_quant: bool = True
    max_memory_usage: float = 0.8
    gpu_timeout_hours: int = 24


@dataclass
class EvaluationConfig:
    """Configuration for evaluation operations."""
    
    models: list = field(default_factory=lambda: ["llama2", "codealpaca", "qwen"])
    metrics: list = field(default_factory=lambda: ["accuracy", "f1", "bleu", "rouge"])
    timeout_seconds: int = 300
    max_samples: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


@dataclass
class JobConfig:
    """Configuration for job execution."""
    
    max_jobs_per_provider: int = 5
    job_timeout_hours: int = 24
    checkpoint_interval: int = 100
    save_checkpoints: bool = True
    max_checkpoint_size_mb: int = 10000
    cleanup_old_checkpoints: bool = True


@dataclass
class SecurityConfig:
    """Security configuration."""
    
    require_authentication: bool = True
    allow_public_models: bool = False
    dataset_validation_required: bool = True
    enable_audit_logging: bool = True
    log_sensitive_data: bool = False
    max_file_upload_size_mb: int = 100
    allowed_file_extensions: list = field(default_factory=lambda: [".json", ".yaml", ".yml", ".txt", ".py"])


@dataclass
class AppConfig:
    """Main application configuration."""
    
    log_level: LogLevel = LogLevel.INFO
    log_format: LogFormat = LogFormat.JSON
    workspace_base: str = "workspace"
    temp_dir: str = "/tmp/nexus"
    cache_dir: str = ".cache/nexus"
    
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    job: JobConfig = field(default_factory=JobConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)


class Config:
    """Configuration manager for NEXUS."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or "config")
        self._config = AppConfig()
        self._loaded_sources = []
        self._load_default_config()
    
    def _load_default_config(self):
        """Load default configuration."""
        self._config.log_level = LogLevel(os.getenv("LOG_LEVEL", "INFO"))
        self._config.log_format = LogFormat(os.getenv("LOG_FORMAT", "json"))
        self._config.workspace_base = os.getenv("WORKSPACE_BASE", "workspace")
        self._config.temp_dir = os.getenv("TEMP_DIR", "/tmp/nexus")
        self._config.cache_dir = os.getenv("CACHE_DIR", ".cache/nexus")
        
        # Create workspace directories
        Path(self._config.workspace_base).mkdir(parents=True, exist_ok=True)
        Path(self._config.temp_dir).mkdir(parents=True, exist_ok=True)
        Path(self._config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def load_from_file(self, file_path: Union[str, Path], source: str):
        """Load configuration from a YAML file."""
        try:
            with open(file_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            self._apply_config(file_config, source)
            self._loaded_sources.append(f"file:{source}")
            
        except Exception as e:
            raise ConfigError(f"Failed to load config from {file_path}: {str(e)}", 
                            config_key=str(file_path))
    
    def load_from_env(self, prefix: str = "NEXUS_"):
        """Load configuration from environment variables."""
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                env_config[config_key] = self._convert_env_value(value)
        
        if env_config:
            self._apply_config(env_config, "environment")
            self._loaded_sources.append("environment")
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable value to appropriate type."""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Handle list values
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        return value
    
    def _apply_config(self, config: Dict[str, Any], source: str):
        """Apply configuration to the internal config object."""
        if not config:
            return
        
        # Recursively apply configuration
        self._apply_nested_config(self._config, config)
    
    def _apply_nested_config(self, target: Any, config: Dict[str, Any]):
        """Recursively apply nested configuration."""
        for key, value in config.items():
            if hasattr(target, key):
                attr = getattr(target, key)
                
                if isinstance(attr, dict) and isinstance(value, dict):
                    # Update the dict
                    attr.update(value)
                elif isinstance(attr, list) and isinstance(value, list):
                    setattr(target, key, value)
                elif hasattr(attr, '__dict__') and isinstance(value, dict):
                    # Recursively apply to object attributes
                    self._apply_nested_config(attr, value)
                else:
                    setattr(target, key, value)
            else:
                # Try to set attribute dynamically
                try:
                    setattr(target, key, value)
                except AttributeError:
                    # If we can't set it, store in a special dict
                    if not hasattr(target, '_extra_config'):
                        target._extra_config = {}
                    target._extra_config[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        current = self._config
        
        for key in keys:
            if hasattr(current, key):
                current = getattr(current, key)
            else:
                return default
        
        return current
    
    def get_all_sources(self) -> list:
        """Get all configuration sources used."""
        return self._loaded_sources.copy()
    
    def validate(self) -> list:
        """Validate current configuration and return list of errors."""
        errors = []
        
        # Validate log level
        if self._config.log_level not in LogLevel:
            errors.append(f"Invalid log level: {self._config.log_level}")
        
        # Validate log format
        if self._config.log_format not in LogFormat:
            errors.append(f"Invalid log format: {self._config.log_format}")
        
        # Validate training config
        if self._config.training.batch_size <= 0:
            errors.append("Training batch size must be positive")
        
        if self._config.training.learning_rate <= 0:
            errors.append("Training learning rate must be positive")
        
        if self._config.training.epochs <= 0:
            errors.append("Training epochs must be positive")
        
        if self._config.training.max_sequence_length <= 0:
            errors.append("Max sequence length must be positive")
        
        # Validate evaluation config
        if not self._config.evaluation.models:
            errors.append("At least one evaluation model must be specified")
        
        if not self._config.evaluation.metrics:
            errors.append("At least one evaluation metric must be specified")
        
        # Validate security config
        if self._config.security.max_file_upload_size_mb <= 0:
            errors.append("Max file upload size must be positive")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config_to_dict(self._config)
    
    def _config_to_dict(self, obj: Any) -> Any:
        """Recursively convert config object to dictionary."""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):
                    result[key] = self._config_to_dict(value)
            return result
        elif isinstance(obj, list):
            return [self._config_to_dict(item) for item in obj]
        else:
            return obj