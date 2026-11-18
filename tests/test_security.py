"""
Security tests for NEXUS.

Tests token redaction, input validation, and security-related functionality.
"""

import os
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch

from nexus.core.logger import RedactionFilter, JSONFormatter
from nexus.core.exceptions import ValidationError, NexusError
from nexus.utils.validation import InputValidator
from nexus.utils.security import SecurityManager


def test_redaction_filter_redacts_tokens():
    """Test that RedactionFilter redacts sensitive token patterns."""
    log_record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='test.py',
        lineno=1,
        msg='Using token hf_1234567890abcdef for authentication',
        args=None,
        exc_info=None
    )
    
    filter_obj = RedactionFilter()
    result = filter_obj.filter(log_record)
    
    assert result is True
    # The message should contain [REDACTED] instead of the token
    assert '[REDACTED]' in log_record.getMessage() or 'hf_1234567890abcdef' not in log_record.getMessage()


def test_redaction_filter_redacts_api_keys():
    """Test that RedactionFilter redacts API key patterns."""
    log_record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='test.py',
        lineno=1,
        msg='API key AKIA1234567890ABCDEF used for request',
        args=None,
        exc_info=None
    )
    
    filter_obj = RedactionFilter()
    filter_obj.filter(log_record)
    
    # Should contain REDACTED pattern
    assert '[REDACTED]' in log_record.getMessage() or 'AKIA' not in log_record.getMessage()


def test_json_formatter_output():
    """Test that JSONFormatter produces valid JSON."""
    formatter = JSONFormatter()
    
    log_record = logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='test.py',
        lineno=42,
        msg='Test log message',
        args=None,
        exc_info=None
    )
    
    output = formatter.format(log_record)
    
    import json
    parsed = json.loads(output)
    assert 'level' in parsed
    assert 'message' in parsed
    assert 'timestamp' in parsed
    assert parsed['message'] == 'Test log message'
    assert parsed['level'] == 'INFO'


def test_input_validator_job_id():
    """Test job ID validation."""
    validator = InputValidator()
    
    # Valid job IDs
    assert validator.validate_job_id('job_123') is True
    assert validator.validate_job_id('job_abc-123_xyz') is True
    
    # Invalid job IDs
    assert validator.validate_job_id('') is False
    assert validator.validate_job_id('job with spaces') is False
    assert validator.validate_job_id('job@#$%') is False


def test_input_validator_parameter_count():
    """Test parameter count validation."""
    validator = InputValidator()
    
    assert validator.validate_parameter_count(500000000) is True  # 500M
    assert validator.validate_parameter_count(0) is False
    assert validator.validate_parameter_count(-1) is False


def test_input_validator_license():
    """Test license validation."""
    validator = InputValidator()
    
    assert validator.validate_license('apache-2.0') is True
    assert validator.validate_license('MIT') is True
    assert validator.validate_license('BSD-3-Clause') is True
    assert validator.validate_license('Proprietary') is False
    assert validator.validate_license('') is False


def test_security_manager_sanitize_path():
    """Test path sanitization to prevent directory traversal."""
    security = SecurityManager()
    
    # Normal paths should work
    safe = security.sanitize_path('workspace/models/model1')
    assert 'workspace' in safe
    
    # Directory traversal should be blocked
    unsafe = security.sanitize_path('../../../etc/passwd')
    assert '..' not in unsafe


def test_security_manager_redact_token():
    """Test that SecurityManager redacts tokens."""
    security = SecurityManager()
    
    text = "Using HF_TOKEN=hf_secret123 for authentication"
    redacted = security.redact_sensitive(text)
    
    assert 'hf_secret123' not in redacted
    assert 'REDACTED' in redacted or '***' in redacted


def test_nexus_error_has_error_code():
    """Test that NexusError includes error codes."""
    error = NexusError("Test error", error_code="TEST_ERROR")
    assert error.error_code == "TEST_ERROR"
    assert "TEST_ERROR" in str(error)


def test_validation_error_has_field():
    """Test that ValidationError includes field info."""
    error = ValidationError("Invalid value", field="model_id", value=None)
    assert error.field == "model_id"
    assert error.value is None