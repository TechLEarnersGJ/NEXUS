"""
Logging infrastructure for NEXUS.

Provides structured logging with support for JSON and text formats,
redaction of sensitive information, and configurable log levels.
"""

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import threading
import os

from nexus.core.exceptions import ValidationError


class RedactionFilter(logging.Filter):
    """Filter to redact sensitive information from log records."""
    
    # Patterns to redact in log messages
    REDACTION_PATTERNS = [
        r'\bhf_[A-Za-z0-9]{8,}\b',  # Hugging Face tokens
        r'\bAKIA[A-Z0-9]{16}\b',  # AWS access keys
        r'\bsk-[A-Za-z0-9]{20,}\b',  # OpenAI-style keys
        r'\bgh_[A-Za-z0-9]{20,}\b',  # GitHub tokens
        r'\bglpat-[A-Za-z0-9]{20,}\b',  # GitLab tokens
        r'\b(?:\d{4}[ -]?){3}\d{4}\b',  # Credit card numbers
        r'\b\d{10,}\b',  # Long numbers (could be IDs/social security)
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
        r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z\b',  # ISO timestamps
    ]
    
    # Environment variable names that contain sensitive data
    SENSITIVE_ENV_VARS = {
        'HF_TOKEN', 'HUGGINGFACE_TOKEN', 'HUGGINGFACEHUB_API_TOKEN',
        'KAGGLE_USERNAME', 'KAGGLE_KEY', 'KAGGLE_API_TOKEN',
        'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY',
        'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
        'AZURE_STORAGE_KEY', 'GITHUB_TOKEN', 'GITLAB_TOKEN'
    }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record, redacting sensitive information."""
        import re
        
        # Get the original message
        message = record.getMessage()
        message_lower = message.lower()
        
        # Redact potential API keys/tokens (simple heuristic)
        # Check for various forms: api_key, api key, token, secret, etc.
        should_redact = (
            'api_key' in message_lower
            or 'api key' in message_lower
            or 'token' in message_lower
            or 'secret' in message_lower
            or 'password' in message_lower
            or 'credential' in message_lower
        )
        
        if should_redact:
            # Replace all token patterns
            redacted = message
            for pattern in self.REDACTION_PATTERNS:
                redacted = re.sub(pattern, '[REDACTED]', redacted)
            
            # Also redact environment variable assignments
            for env_var in self.SENSITIVE_ENV_VARS:
                redacted = re.sub(
                    rf'{env_var}\s*=\s*[^\s"\']+',
                    f'{env_var}=[REDACTED]',
                    redacted,
                    flags=re.IGNORECASE,
                )
            
            # Update msg so getMessage() returns redacted version
            record.msg = redacted
        
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_level: bool = True, include_timestamp: bool = True,
                 include_thread: bool = True, include_process: bool = True):
        super().__init__()
        self.include_level = include_level
        self.include_timestamp = include_timestamp
        self.include_thread = include_thread
        self.include_process = include_process
        self.thread_local = threading.local()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Prepare log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "level": record.levelname,
            "message": record.getMessage(),
            "logger_name": record.name,
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        # Add optional fields
        if self.include_thread and hasattr(record, 'thread'):
            log_entry["thread"] = record.thread
        
        if self.include_process and hasattr(record, 'process'):
            log_entry["process"] = record.process
        
        # Add exception info if present
        if record.exc_info:
            import traceback
            log_entry["exception"] = traceback.format_exception(*record.exc_info)
        
        # Add extra fields from record
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Text formatter for human-readable logging."""
    
    def __init__(self, format_string: Optional[str] = None):
        if format_string is None:
            format_string = (
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(message)s'
            )
        super().__init__(format_string, datefmt='%Y-%m-%d %H:%M:%S')


class Logger:
    """Custom logger for NEXUS with enhanced features."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._loggers = {}
        self._config = {}
    
    def configure(self, log_dir: str, log_level: str = "INFO", 
                  log_format: str = "json", max_file_size_mb: int = 100,
                  backup_count: int = 5):
        """Configure the logger system."""
        # Convert enums to strings if needed
        if hasattr(log_level, 'value'):
            log_level = log_level.value
        elif hasattr(log_level, 'name'):
            log_level = log_level.name
        else:
            log_level = str(log_level)
            
        if hasattr(log_format, 'value'):
            log_format = log_format.value
        elif hasattr(log_format, 'name'):
            log_format = log_format.name
        else:
            log_format = str(log_format)
        
        self._config = {
            'log_dir': log_dir,
            'log_level': log_level,
            'log_format': log_format,
            'max_file_size_mb': max_file_size_mb,
            'backup_count': backup_count
        }
        
        # Ensure log directory exists
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.getLevelName(log_level))
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create handler
        if log_format.lower() == "json":
            formatter = JSONFormatter()
        else:
            formatter = TextFormatter()
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, 'nexus.log'),
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RedactionFilter())
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Add handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Get our logger
        self._logger = self.get_logger(__name__)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def log_event(self, event_type: str, details: Dict[str, Any],
                  logger_name: str = "nexus"):
        """Log a structured event."""
        logger = self.get_logger(logger_name)
        
        # Add event type to details
        log_data = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "details": details
        }
        
        # Create extra fields for structured logging
        extra_fields = {k: v for k, v in log_data.items()}
        
        logger.info("Event logged", extra=extra_fields)


# Global logger instance
logger = Logger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name from the global logger instance."""
    return logger.get_logger(name)