"""
Tests for NEXUS base trainer.
"""

import tempfile
from pathlib import Path

from nexus.training.trainer import BaseTrainer


def test_trainer_initialization():
    """Test trainer initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = BaseTrainer(config={
            'output_dir': tmpdir,
        })
        
        assert trainer.current_epoch == 0
        assert trainer.current_step == 0
        assert trainer.total_steps == 0
        assert trainer.best_loss == float('inf')
        assert trainer.training_history == []
        
        # Check directories were created
        tmpdir_path = Path(tmpdir)
        assert tmpdir_path.exists()
        assert (tmpdir_path / 'checkpoints').exists()
        assert (tmpdir_path / 'logs').exists()


def test_trainer_training_loop():
    """Test basic training loop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = BaseTrainer(config={
            'output_dir': tmpdir,
        })
        
        result = trainer.train(
            model_id='test-model',
            dataset_id='test-dataset',
            train_config={
                'job_id': 'test_job',
                'num_epochs': 2,
                'batch_size': 2,
                'gradient_accumulation_steps': 4,
                'learning_rate': 0.0001,
                'max_sequence_length': 2048,
                'logging_steps': 10,
                'save_strategy': 'epoch',
            },
        )
        
        assert result['status'] == 'completed'
        assert result['model_id'] == 'test-model'
        assert result['dataset_id'] == 'test-dataset'
        assert result['epochs_completed'] == 2
        assert result['total_steps'] > 0
        assert result['final_loss'] is not None
        assert result['training_time_seconds'] > 0
        assert len(result['training_history']) == 2


def test_trainer_checkpoint_save_load():
    """Test trainer checkpoint save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = BaseTrainer(config={
            'output_dir': tmpdir,
        })
        
        # Run training to create checkpoints
        trainer.train(
            model_id='test-model',
            dataset_id='test-dataset',
            train_config={
                'job_id': 'test_job',
                'num_epochs': 2,
                'batch_size': 2,
                'save_strategy': 'epoch',
            },
        )
        
        # List checkpoints
        checkpoints = trainer.list_checkpoints()
        assert len(checkpoints) == 2  # One per epoch
        
        # Check checkpoint structure
        for cp in checkpoints:
            assert 'path' in cp
            assert 'epoch' in cp
            assert 'step' in cp
            assert Path(cp['path']).exists()


def test_trainer_training_summary():
    """Test training summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = BaseTrainer(config={
            'output_dir': tmpdir,
        })
        
        # Initial summary
        summary = trainer.get_training_summary()
        assert summary['current_epoch'] == 0
        assert summary['current_step'] == 0
        assert summary['checkpoints'] == []
        
        # Run training
        trainer.train(
            model_id='test-model',
            dataset_id='test-dataset',
            train_config={
                'job_id': 'test_job',
                'num_epochs': 1,
                'batch_size': 2,
                'save_strategy': 'epoch',
            },
        )
        
        # Summary after training
        summary = trainer.get_training_summary()
        assert summary['current_epoch'] == 1
        assert summary['current_step'] > 0
        assert len(summary['checkpoints']) == 1