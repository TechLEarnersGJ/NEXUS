"""
Tests for NEXUS configuration management.
"""

import os
import tempfile
import yaml
from pathlib import Path

from nexus.core.config import Config, LogLevel, LogFormat, AppConfig


def test_config_default_values():
    """Test that Config uses default values when no config files provided."""
    config = Config()
    
    assert config.get('log_level') == LogLevel.INFO
    assert config.get('log_format') == LogFormat.JSON
    assert config.get('workspace_base') == 'workspace'
    assert config.get('temp_dir') == '/tmp/nexus'
    assert config.get('cache_dir') == '.cache/nexus'


def test_config_validation_passes():
    """Test that default configuration passes validation."""
    config = Config()
    errors = config.validate()
    assert len(errors) == 0, f"Config validation failed: {errors}"


def test_config_load_from_file(tmp_path):
    """Test loading configuration from a YAML file."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    config_file = config_dir / "settings.yaml"
    config_data = {
        'log_level': 'DEBUG',
        'log_format': 'text',
        'workspace_base': 'test_workspace',
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    config = Config()
    config.load_from_file(str(config_file), 'settings')
    
    # Config.get returns enum objects, compare by value
    log_level = config.get('log_level')
    assert log_level.value == 'DEBUG' if hasattr(log_level, 'value') else log_level == 'DEBUG'
    log_format = config.get('log_format')
    assert log_format.value == 'text' if hasattr(log_format, 'value') else log_format == 'text'
    assert config.get('workspace_base') == 'test_workspace'


def test_config_validation_errors():
    """Test that invalid configuration is caught."""
    config = Config()
    
    # Set invalid values
    config._config.log_level = 'INVALID'
    errors = config.validate()
    
    assert len(errors) > 0
    assert any('log level' in e.lower() for e in errors)


def test_config_get_all_sources():
    """Test that configuration sources are tracked."""
    config = Config()
    sources = config.get_all_sources()
    assert isinstance(sources, list)


def test_config_to_dict():
    """Test configuration serialization to dictionary."""
    config = Config()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert 'log_level' in config_dict
    assert 'workspace_base' in config_dict


def test_env_var_redaction():
    """Test that sensitive environment variables are not exposed in config."""
    # This test ensures tokens never leak through config
    config = Config()
    config_dict = config.to_dict()
    
    # Check no token-like values in config
    config_str = str(config_dict)
    assert 'hf_' not in config_str.lower(), "HF token found in config dict!"
    assert 'sk-' not in config_str.lower(), "API key found in config dict!"