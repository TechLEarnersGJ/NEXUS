"""
Input validation utilities for NEXUS.

Provides validation for job IDs, model metadata, licenses, and other inputs.
"""

import re
from typing import Any, Optional


class InputValidator:
    """Input validation for NEXUS."""
    
    # Valid characters for job IDs
    JOB_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    
    # Valid license identifiers
    ALLOWED_LICENSES = {
        'mit', 'apache-2.0', 'apache2', 'bsd-3-clause', 'bsd-2-clause',
        'cc-by-4.0', 'cc-by-sa-4.0', 'llama2', 'gpl-3.0', 'lgpl-3.0',
        'mpl-2.0', 'unlicense', 'cc0-1.0',
    }
    
    def validate_job_id(self, job_id: str) -> bool:
        """Validate a job ID string."""
        if not job_id or not isinstance(job_id, str):
            return False
        
        if len(job_id) < 1 or len(job_id) > 100:
            return False
        
        return bool(self.JOB_ID_PATTERN.match(job_id))
    
    def validate_parameter_count(self, count: Any) -> bool:
        """Validate model parameter count."""
        if not isinstance(count, (int, float)):
            return False
        
        return count > 0
    
    def validate_license(self, license_str: str) -> bool:
        """Validate a license string."""
        if not license_str or not isinstance(license_str, str):
            return False
        
        return license_str.lower() in self.ALLOWED_LICENSES
    
    def validate_model_id(self, model_id: str) -> bool:
        """Validate a Hugging Face model ID format."""
        if not model_id or not isinstance(model_id, str):
            return False
        
        # Hugging Face model IDs are typically "author/model-name"
        parts = model_id.split('/')
        if len(parts) != 2:
            return False
        
        author, name = parts
        if not author or not name:
            return False
        
        # Check for invalid characters
        if re.search(r'[<>:"/\\|?*\x00-\x1f]', model_id):
            return False
        
        return True
    
    def validate_dataset_id(self, dataset_id: str) -> bool:
        """Validate a Hugging Face dataset ID format."""
        return self.validate_model_id(dataset_id)
    
    def validate_url(self, url: str) -> bool:
        """Validate a URL string."""
        if not url or not isinstance(url, str):
            return False
        
        return url.startswith(('http://', 'https://'))
    
    def validate_email(self, email: str) -> bool:
        """Validate an email address."""
        if not email or not isinstance(email, str):
            return False
        
        pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        return bool(pattern.match(email))
    
    def sanitize_string(self, value: str, max_length: int = 255) -> str:
        """Sanitize a string for safe use."""
        if not isinstance(value, str):
            return str(value)
        
        # Remove null bytes and control characters
        sanitized = ''.join(c for c in value if ord(c) >= 32 or c in '\t\n\r')
        
        # Truncate to max length
        return sanitized[:max_length]
    
    def validate_training_config(self, config: dict) -> list:
        """Validate training configuration and return list of errors."""
        errors = []
        
        if not isinstance(config, dict):
            errors.append("Config must be a dictionary")
            return errors
        
        # Validate batch size
        batch_size = config.get('batch_size', 2)
        if not isinstance(batch_size, int) or batch_size < 1:
            errors.append("batch_size must be a positive integer")
        
        # Validate learning rate
        lr = config.get('learning_rate', 0.0001)
        if not isinstance(lr, (int, float)) or lr <= 0:
            errors.append("learning_rate must be a positive number")
        
        # Validate epochs
        epochs = config.get('epochs', 3)
        if not isinstance(epochs, int) or epochs < 1:
            errors.append("epochs must be a positive integer")
        
        # Validate max sequence length
        max_seq = config.get('max_sequence_length', 2048)
        if not isinstance(max_seq, int) or max_seq < 1:
            errors.append("max_sequence_length must be a positive integer")
        
        return errors