"""
Tests for NEXUS LoRA and QLoRA trainers.
"""

from nexus.training.lora import LoRATrainer, LoRAConfig
from nexus.training.qlora import QLoRATrainer, QLoRAConfig


def test_lora_config_creation():
    """Test LoRA configuration creation."""
    trainer = LoRATrainer()
    config = trainer.create_config(
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
    )
    
    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.learning_rate == 0.0001
    assert config.batch_size == 2
    assert config.num_epochs == 3
    assert config.max_sequence_length == 2048
    assert config.gradient_checkpointing is True
    assert config.mixed_precision == 'bfloat16'


def test_lora_config_with_overrides():
    """Test LoRA config with custom overrides."""
    trainer = LoRATrainer()
    config = trainer.create_config(
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
        method_overrides={
            'r': 32,
            'lora_alpha': 64,
            'learning_rate': 0.0002,
            'batch_size': 4,
        },
    )
    
    assert config.r == 32
    assert config.lora_alpha == 64
    assert config.learning_rate == 0.0002
    assert config.batch_size == 4


def test_lora_config_to_dict():
    """Test LoRA config serialization."""
    trainer = LoRATrainer()
    config = trainer.create_config(
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
    )
    d = config.to_dict()
    
    assert d['r'] == 16
    assert d['lora_alpha'] == 32
    assert 'target_modules' in d
    assert 'task_type' in d


def test_lora_job_spec():
    """Test LoRA job spec creation."""
    trainer = LoRATrainer()
    config = trainer.create_config(
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
    )
    
    job_spec = trainer.create_job_spec(
        job_id='test_job_123',
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
        config=config,
        provider='kaggle',
    )
    
    assert job_spec['job_id'] == 'test_job_123'
    assert job_spec['method'] == 'lora'
    assert job_spec['provider'] == 'kaggle'
    assert job_spec['config']['base_model_id'] == 'Qwen/Qwen2.5-0.5B-Instruct'
    assert job_spec['config']['dataset_id'] == 'test/dataset'
    assert 'lora_config' in job_spec['config']


def test_lora_recommended_settings():
    """Test LoRA recommended settings."""
    trainer = LoRATrainer()
    
    # Small model on T4 (14.56GB < 16GB, so batch_size is halved from 4 to 2)
    settings = trainer.get_recommended_settings(
        model_parameter_count=500_000_000,
        gpu_memory_gb=14.56,
    )
    assert settings['r'] == 16
    assert settings['batch_size'] == 2  # Halved due to T4 memory < 16GB
    
    # Same model on A100 (40GB >= 16GB, batch stays at 4)
    settings_a100 = trainer.get_recommended_settings(
        model_parameter_count=500_000_000,
        gpu_memory_gb=40.0,
    )
    assert settings_a100['batch_size'] == 4
    
    # Large model
    settings = trainer.get_recommended_settings(
        model_parameter_count=70_000_000_000,
        gpu_memory_gb=14.56,
    )
    assert settings['r'] == 8
    assert settings['batch_size'] == 1  # Large model needs smaller batch


def test_qlora_config_creation():
    """Test QLoRA configuration creation."""
    trainer = QLoRATrainer()
    config = trainer.create_config(
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
    )
    
    assert config.bits == 4
    assert config.quant_type == 'nf4'
    assert config.compute_dtype == 'bfloat16'
    assert config.double_quant is True
    assert config.r == 16
    assert config.lora_alpha == 32


def test_qlora_config_with_overrides():
    """Test QLoRA config with custom overrides."""
    trainer = QLoRATrainer()
    config = trainer.create_config(
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
        method_overrides={
            'r': 8,
            'bits': 8,
            'batch_size': 1,
        },
    )
    
    assert config.r == 8
    assert config.bits == 8
    assert config.batch_size == 1


def test_qlora_config_to_dict():
    """Test QLoRA config serialization."""
    trainer = QLoRATrainer()
    config = trainer.create_config(
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
    )
    d = config.to_dict()
    
    assert d['bits'] == 4
    assert d['quant_type'] == 'nf4'
    assert d['double_quant'] is True
    assert 'r' in d


def test_qlora_job_spec():
    """Test QLoRA job spec creation."""
    trainer = QLoRATrainer()
    config = trainer.create_config(
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
    )
    
    job_spec = trainer.create_job_spec(
        job_id='test_qlora_job',
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
        config=config,
        provider='colab',
    )
    
    assert job_spec['job_id'] == 'test_qlora_job'
    assert job_spec['method'] == 'qlora'
    assert job_spec['provider'] == 'colab'
    assert job_spec['config']['base_model_id'] == 'Qwen/Qwen2.5-0.5B-Instruct'
    assert 'qlora_config' in job_spec['config']


def test_qlora_recommended_settings():
    """Test QLoRA recommended settings."""
    trainer = QLoRATrainer()
    
    settings = trainer.get_recommended_settings(
        model_parameter_count=500_000_000,
        gpu_memory_gb=14.56,
    )
    assert settings['bits'] == 4
    assert settings['quant_type'] == 'nf4'
    assert settings['double_quant'] is True
    assert settings['r'] == 16