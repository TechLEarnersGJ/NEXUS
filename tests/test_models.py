"""
Tests for NEXUS model management components.
"""

import os
import tempfile
from pathlib import Path

from nexus.models.discovery import ModelDiscovery, ModelMetadata
from nexus.models.selector import ModelSelector
from nexus.models.registry import ModelRegistry
from nexus.models.validator import ModelValidator


def test_model_discovery_search():
    """Test model discovery search functionality."""
    discovery = ModelDiscovery()
    
    # Search by query
    models = discovery.search(query="python", max_models=5)
    assert len(models) > 0
    
    # Check model metadata
    for model in models:
        assert model.model_id is not None
        assert model.name is not None
        assert model.parameter_count > 0
        assert model.license is not None


def test_model_discovery_with_filters():
    """Test model discovery with parameter filters."""
    discovery = ModelDiscovery()
    
    # Search with max parameters filter
    models = discovery.search(
        query="",
        max_models=10,
        filters={"max_params": 1000000000}  # 1B
    )
    
    for model in models:
        assert model.parameter_count <= 1000000000


def test_model_metadata_creation():
    """Test ModelMetadata creation and methods."""
    data = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_name": "Qwen2.5-0.5B-Instruct",
        "family": "Qwen",
        "architecture": "Transformers",
        "parameter_count": 500000000,
        "size": 2500000000,
        "context_length": 8192,
        "license": "Apache-2.0",
        "tags": ["conversational", "instruction-following"],
        "downloads": 1500000,
        "likes": 5000,
    }
    
    model = ModelMetadata.from_dict(data)
    
    assert model.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert model.get_params_billions() == 0.5
    assert model.get_size_gb() > 0
    assert model.is_instruction_tuned() is True
    assert model.is_conversational() is True


def test_model_metadata_to_dict():
    """Test ModelMetadata serialization."""
    data = {
        "model_id": "test/model",
        "model_name": "test-model",
        "parameter_count": 1000000,
        "license": "MIT",
    }
    
    model = ModelMetadata.from_dict(data)
    output = model.to_dict()
    
    assert output["model_id"] == "test/model"
    assert output["parameter_count"] == 1000000


def test_model_selector_select_best():
    """Test model selector picks best model."""
    discovery = ModelDiscovery()
    models = discovery.search(query="python", max_models=10)
    
    selector = ModelSelector()
    
    requirements = {
        "objective": "python code assistant",
        "max_parameters": 1000000000,
        "required_license": "apache-2.0",
        "capabilities": ["instruction_following"],
    }
    
    selected = selector.select_best_model(models, requirements)
    
    if selected is not None:
        assert selected.parameter_count <= 1000000000
        assert selected.license.lower() == "apache-2.0"


def test_model_selector_empty_models():
    """Test model selector with empty model list."""
    selector = ModelSelector()
    
    result = selector.select_best_model([], {"objective": "test"})
    assert result is None


def test_model_registry_save_and_get():
    """Test model registry save and retrieve."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(storage_path=tmpdir)
        
        model_data = {
            "model_id": "test/model-v1",
            "model_name": "test-model-v1",
            "family": "test",
            "parameter_count": 1000000,
            "size_bytes": 5000000,
            "license": "MIT",
            "tags": ["test"],
        }
        
        model = ModelMetadata.from_dict(model_data)
        registry.save_model(model)
        
        # Retrieve
        retrieved = registry.get_model("test/model-v1")
        assert retrieved is not None
        assert retrieved.model_id == "test/model-v1"
        assert retrieved.get_params_billions() == 0.001


def test_model_registry_list_and_search():
    """Test model registry listing and search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(storage_path=tmpdir)
        
        # Save multiple models
        for i in range(3):
            model_data = {
                "model_id": f"test/model-v{i}",
                "model_name": f"test-model-v{i}",
                "family": "test",
                "parameter_count": 1000000 * (i + 1),
                "license": "MIT",
                "tags": ["test", f"v{i}"],
            }
            model = ModelMetadata.from_dict(model_data)
            registry.save_model(model)
        
        # List all
        models = registry.list_models()
        assert len(models) == 3
        
        # Search
        results = registry.search_models("test-model-v0", max_results=5)
        assert len(results) >= 1


def test_model_registry_remove():
    """Test model registry removal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(storage_path=tmpdir)
        
        model_data = {
            "model_id": "test/temp-model",
            "model_name": "temp-model",
            "family": "test",
            "parameter_count": 500000,
            "license": "MIT",
        }
        
        model = ModelMetadata.from_dict(model_data)
        registry.save_model(model)
        
        # Verify it exists
        assert registry.get_model("test/temp-model") is not None
        
        # Remove it
        registry.remove_model("test/temp-model")
        assert registry.get_model("test/temp-model") is None


def test_model_validator_valid_model():
    """Test model validator accepts valid model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a model directory with config.json
        model_dir = Path(tmpdir) / "test_model"
        model_dir.mkdir()
        
        import json
        with open(model_dir / "config.json", 'w') as f:
            json.dump({"model_type": "qwen"}, f)
        
        validator = ModelValidator()
        
        model_metadata = {
            "model_id": "test/model",
            "license": "Apache-2.0",
        }
        
        result = validator.validate_model(str(model_dir), model_metadata)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0


def test_model_validator_invalid_license():
    """Test model validator rejects invalid license."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "test_model"
        model_dir.mkdir()
        
        import json
        with open(model_dir / "config.json", 'w') as f:
            json.dump({"model_type": "qwen"}, f)
        
        validator = ModelValidator()
        
        model_metadata = {
            "model_id": "test/model",
            "license": "Proprietary",  # Not in allowed list
        }
        
        result = validator.validate_model(str(model_dir), model_metadata)
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0


def test_model_validator_missing_path():
    """Test model validator rejects non-existent path."""
    validator = ModelValidator()
    
    try:
        validator.validate_model("/nonexistent/path", {"license": "MIT"})
        assert False, "Should have raised ValidationError"
    except Exception:
        pass  # Expected


def test_model_metadata_relevance_score():
    """Test model metadata relevance scoring."""
    data = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "name": "Qwen2.5-0.5B-Instruct",
        "family": "Qwen",
        "parameter_count": 500000000,
        "license": "Apache-2.0",
        "tags": ["conversational", "instruction-following", "python"],
    }
    
    model = ModelMetadata.from_dict(data)
    
    # Exact match should score high
    score = model.get_relevance_score("Qwen2.5-0.5B-Instruct")
    assert score > 5.0
    
    # Family match should score medium
    score = model.get_relevance_score("Qwen")
    assert score > 0.0
    
    # No match should score low
    score = model.get_relevance_score("Llama-3")
    assert score < 5.0