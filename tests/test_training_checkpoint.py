"""
Tests for NEXUS checkpoint manager.
"""

import json
import tempfile
from pathlib import Path

from nexus.training.checkpoint import CheckpointManager


def test_checkpoint_save_and_load():
    """Test checkpoint save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(config={'checkpoint_dir': tmpdir})
        
        # Save checkpoint
        result = mgr.save_checkpoint(
            job_id='test_job',
            epoch=1,
            step=100,
            loss=0.5,
            model_path=None,
            adapter_path=None,
            metadata={'model': 'Qwen2.5-0.5B'},
        )
        
        assert result['saved'] is True
        assert result['epoch'] == 1
        assert result['step'] == 100
        assert result['loss'] == 0.5
        assert 'checkpoint_path' in result
        
        # Load checkpoint
        loaded = mgr.load_checkpoint(result['checkpoint_path'])
        assert loaded['job_id'] == 'test_job'
        assert loaded['epoch'] == 1
        assert loaded['step'] == 100
        assert loaded['loss'] == 0.5


def test_checkpoint_list():
    """Test checkpoint listing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(config={'checkpoint_dir': tmpdir})
        
        # Initially empty
        checkpoints = mgr.list_checkpoints()
        assert len(checkpoints) == 0
        
        # Save multiple checkpoints
        for i in range(3):
            mgr.save_checkpoint(
                job_id='test_job',
                epoch=i + 1,
                step=(i + 1) * 100,
                loss=1.0 - (i * 0.1),
            )
        
        # List all
        checkpoints = mgr.list_checkpoints()
        assert len(checkpoints) == 3
        
        # Filter by job
        checkpoints = mgr.list_checkpoints(job_id='test_job')
        assert len(checkpoints) == 3
        
        # Different job should not appear
        checkpoints = mgr.list_checkpoints(job_id='other_job')
        assert len(checkpoints) == 0


def test_checkpoint_latest():
    """Test getting latest checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(config={'checkpoint_dir': tmpdir})
        
        # No checkpoints
        latest = mgr.get_latest_checkpoint('test_job')
        assert latest is None
        
        # Save checkpoints
        mgr.save_checkpoint(job_id='test_job', epoch=1, step=100, loss=0.8)
        mgr.save_checkpoint(job_id='test_job', epoch=2, step=200, loss=0.5)
        mgr.save_checkpoint(job_id='test_job', epoch=3, step=300, loss=0.3)
        
        latest = mgr.get_latest_checkpoint('test_job')
        assert latest is not None
        assert latest['epoch'] == 3
        assert latest['step'] == 300
        assert latest['loss'] == 0.3


def test_checkpoint_delete():
    """Test checkpoint deletion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(config={'checkpoint_dir': tmpdir})
        
        result = mgr.save_checkpoint(
            job_id='test_job', epoch=1, step=100, loss=0.5,
        )
        checkpoint_path = result['checkpoint_path']
        
        # Verify it exists
        assert Path(checkpoint_path).exists()
        
        # Delete it
        deleted = mgr.delete_checkpoint(checkpoint_path)
        assert deleted is True
        assert not Path(checkpoint_path).exists()
        
        # Delete non-existent
        deleted = mgr.delete_checkpoint('/nonexistent/path')
        assert deleted is False


def test_checkpoint_cleanup():
    """Test old checkpoint cleanup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(config={
            'checkpoint_dir': tmpdir,
            'max_checkpoints': 2,
            'cleanup_old_checkpoints': True,
        })
        
        # Save 5 checkpoints
        for i in range(5):
            mgr.save_checkpoint(
                job_id='test_job',
                epoch=i + 1,
                step=(i + 1) * 100,
                loss=1.0 - (i * 0.1),
            )
        
        # Should only keep 2
        checkpoints = mgr.list_checkpoints(job_id='test_job')
        assert len(checkpoints) == 2


def test_checkpoint_stats():
    """Test checkpoint statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(config={'checkpoint_dir': tmpdir})
        
        mgr.save_checkpoint(job_id='job1', epoch=1, step=100, loss=0.5)
        mgr.save_checkpoint(job_id='job2', epoch=1, step=100, loss=0.6)
        
        stats = mgr.get_checkpoint_stats()
        assert stats['total_checkpoints'] == 2
        assert stats['jobs_with_checkpoints'] == 2
        assert 'total_size_mb' in stats
        assert 'checkpoint_dir' in stats


def test_checkpoint_checkpoint_meta_file():
    """Test that checkpoint meta file is valid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(config={'checkpoint_dir': tmpdir})
        
        result = mgr.save_checkpoint(
            job_id='test_job', epoch=1, step=50, loss=0.75,
        )
        
        meta_file = Path(result['checkpoint_path']) / 'checkpoint_meta.json'
        assert meta_file.exists()
        
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        assert meta['job_id'] == 'test_job'
        assert meta['epoch'] == 1
        assert meta['step'] == 50
        assert meta['loss'] == 0.75
        assert 'saved_at' in meta