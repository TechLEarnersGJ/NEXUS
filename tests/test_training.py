"""
Tests for NEXUS training engine.
"""

from nexus.core.config import Config
from nexus.training.engine import TrainingEngine


def test_training_engine_initialization():
    """Test training engine initialization."""
    config = Config()
    engine = TrainingEngine(config)
    
    assert engine.default_method == "qlora"
    assert engine.default_batch_size == 2
    assert engine.default_learning_rate == 0.0001
    assert engine.default_epochs == 3


def test_training_engine_create_job():
    """Test training engine job creation."""
    config = Config()
    engine = TrainingEngine(config)
    
    job_spec = {
        "job_id": "train_job_1",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset": "python-code-instructions",
        "method": "qlora",
        "batch_size": 2,
        "learning_rate": 0.0001,
        "epochs": 3,
    }
    
    job = engine.create_training_job(job_spec)
    
    assert job["job_id"] == "train_job_1"
    assert job["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert job["method"] == "qlora"
    assert job["status"] == "pending"
    assert job["progress"] == 0.0


def test_training_engine_get_methods():
    """Test getting available training methods."""
    config = Config()
    engine = TrainingEngine(config)
    
    methods = engine.get_training_methods()
    
    assert "sft" in methods
    assert "lora" in methods
    assert "qlora" in methods
    assert "peft" in methods


def test_training_engine_get_recommended_settings():
    """Test getting recommended settings for model sizes."""
    config = Config()
    engine = TrainingEngine(config)
    
    # Small model
    settings = engine.get_recommended_settings("Qwen2.5-0.5B")
    assert settings["method"] == "lora"
    assert settings["batch_size"] == 2
    
    # Medium model
    settings = engine.get_recommended_settings("Llama-2-7B")
    assert settings["method"] == "qlora"
    
    # Large model
    settings = engine.get_recommended_settings("Llama-2-70B")
    assert settings["method"] == "qlora"


def test_training_engine_estimate_time():
    """Test training time estimation."""
    config = Config()
    engine = TrainingEngine(config)
    
    job_spec = {
        "dataset_size": 100000,
        "model_parameters": 500000000,
    }
    
    time_estimate = engine._estimate_training_time(job_spec)
    assert time_estimate > 0
    assert time_estimate <= 7200.0  # Cap at 2 hours


def test_training_engine_estimate_memory():
    """Test GPU memory estimation."""
    config = Config()
    engine = TrainingEngine(config)
    
    job_spec = {
        "model_parameters": 500000000,
        "batch_size": 2,
    }
    
    memory_estimate = engine._estimate_gpu_memory(job_spec)
    assert memory_estimate > 0
    assert memory_estimate <= 24.0  # Cap at 24 GB