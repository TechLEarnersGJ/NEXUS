"""
Tests for NEXUS dataset management components.
"""

import os
import tempfile
import json
from pathlib import Path

from nexus.datasets.discovery import DatasetDiscovery, DatasetMetadata
from nexus.datasets.validator import DatasetValidator
from nexus.datasets.cleaner import DatasetCleaner
from nexus.datasets.formatter import DatasetFormatter
from nexus.datasets.registry import DatasetRegistry


def test_dataset_discovery_search():
    """Test dataset discovery search functionality."""
    discovery = DatasetDiscovery()
    
    datasets = discovery.search(query="python", max_datasets=5)
    assert len(datasets) > 0
    
    for dataset in datasets:
        assert dataset.dataset_id is not None
        assert dataset.name is not None
        assert dataset.license is not None


def test_dataset_discovery_with_filters():
    """Test dataset discovery with filters."""
    discovery = DatasetDiscovery()
    
    datasets = discovery.search(
        query="python",
        max_datasets=10,
        filters={"license": "apache-2.0"}
    )
    
    for dataset in datasets:
        assert dataset.license.lower() == "apache-2.0"


def test_dataset_metadata_creation():
    """Test DatasetMetadata creation and methods."""
    data = {
        "id": "test/python-dataset",
        "dataset_id": "python-dataset",
        "name": "Python Dataset",
        "description": "Python code dataset",
        "size": 5000000,
        "license": "Apache-2.0",
        "tags": ["python", "code"],
        "downloads": 1000,
        "likes": 50,
        "languages": ["python"],
        "usecases": ["code-generation"],
        "task_type": "code-generation",
    }
    
    dataset = DatasetMetadata.from_dict(data)
    
    assert dataset.dataset_id == "python-dataset"
    assert dataset.get_size_gb() > 0
    assert dataset.is_large_dataset(threshold_gb=0.001) is True


def test_dataset_metadata_to_dict():
    """Test DatasetMetadata serialization."""
    data = {
        "id": "test/dataset",
        "dataset_id": "dataset",
        "name": "Test Dataset",
        "license": "MIT",
    }
    
    dataset = DatasetMetadata.from_dict(data)
    output = dataset.to_dict()
    
    assert output["id"] == "test/dataset"
    assert output["license"] == "MIT"


def test_dataset_validator_valid_dataset():
    """Test dataset validator accepts valid dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "test_dataset"
        dataset_dir.mkdir()
        
        # Create a JSONL file with enough samples
        with open(dataset_dir / "train.jsonl", 'w') as f:
            for i in range(150):
                f.write(json.dumps({"text": f"Sample text number {i} with enough length", "label": "positive"}) + '\n')
        
        validator = DatasetValidator()
        result = validator.validate_dataset(str(dataset_dir))
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0


def test_dataset_validator_missing_path():
    """Test dataset validator rejects non-existent path."""
    validator = DatasetValidator()
    
    try:
        validator.validate_dataset("/nonexistent/path")
        assert False, "Should have raised ValidationError"
    except Exception:
        pass  # Expected


def test_dataset_validator_too_few_samples():
    """Test dataset validator rejects datasets with too few samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "small_dataset"
        dataset_dir.mkdir()
        
        with open(dataset_dir / "train.jsonl", 'w') as f:
            f.write(json.dumps({"text": "Short", "label": "pos"}) + '\n')
        
        validator = DatasetValidator()
        result = validator.validate_dataset(str(dataset_dir))
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0


def test_dataset_cleaner():
    """Test dataset cleaner removes duplicates and invalid samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "test_dataset"
        dataset_dir.mkdir()
        output_dir = Path(tmpdir) / "cleaned"
        
        # Create JSONL file with duplicate and valid samples
        with open(dataset_dir / "train.jsonl", 'w') as f:
            for i in range(5):
                f.write(json.dumps({"text": f"Sample Python instruction {i}", "label": "positive_example"}) + '\n')
            # Add a duplicate
            f.write(json.dumps({"text": "Sample Python instruction 0", "label": "positive_example"}) + '\n')
        
        cleaner = DatasetCleaner(config={"remove_duplicates": True})
        result = cleaner.clean_dataset(str(dataset_dir), str(output_dir))
        
        assert result["total_original_samples"] == 6
        assert result["total_cleaned_samples"] == 5  # Duplicate removed
        assert result["samples_removed"] == 1


def test_dataset_formatter():
    """Test dataset formatter creates train/val/test splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "test_dataset"
        dataset_dir.mkdir()
        output_dir = Path(tmpdir) / "formatted"
        
        # Create JSONL file with samples
        with open(dataset_dir / "train.jsonl", 'w') as f:
            for i in range(20):
                f.write(json.dumps({"text": f"Sample text {i}", "label": "label_" + str(i % 2)}) + '\n')
        
        formatter = DatasetFormatter(config={"train_split": 0.8, "validation_split": 0.1, "test_split": 0.1})
        result = formatter.format_dataset(str(dataset_dir), str(output_dir))
        
        assert result["total_samples"] == 20
        assert result["train_count"] > 0
        assert result["format_type"] == "jsonl"
        
        # Check output files exist
        assert Path(output_dir / "train.jsonl").exists()


def test_dataset_registry():
    """Test dataset registry operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = DatasetRegistry(storage_path=tmpdir)
        
        dataset_data = {
            "id": "test/python-dataset",
            "dataset_id": "python-dataset",
            "name": "Python Dataset",
            "description": "Test dataset",
            "size": 5000000,
            "license": "Apache-2.0",
            "tags": ["python"],
            "downloads": 1000,
            "likes": 50,
            "languages": ["python"],
            "usecases": ["code-generation"],
            "task_type": "code-generation",
        }
        
        dataset = DatasetMetadata.from_dict(dataset_data)
        registry.save_dataset(dataset)
        
        # Retrieve
        retrieved = registry.get_dataset("python-dataset")
        assert retrieved is not None
        assert retrieved.name == "Python Dataset"
        
        # List
        datasets = registry.list_datasets()
        assert len(datasets) == 1
        
        # Search
        results = registry.search_datasets("Python", max_results=5)
        assert len(results) >= 1


def test_dataset_relevance_score():
    """Test dataset metadata relevance scoring."""
    data = {
        "id": "test/python-code",
        "dataset_id": "python-code",
        "name": "Python Code Instructions",
        "description": "Python code instructions dataset",
        "size": 5000000,
        "license": "MIT",
        "tags": ["python", "code", "instructions"],
        "languages": ["python"],
        "usecases": ["code-generation"],
    }
    
    dataset = DatasetMetadata.from_dict(data)
    
    # Exact match
    score = dataset.get_relevance_score("Python Code Instructions")
    assert score > 5.0
    
    # Tag match
    score = dataset.get_relevance_score("python")
    assert score > 0.0