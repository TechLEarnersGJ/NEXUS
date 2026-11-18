"""
Checkpoint manager for NEXUS.

Manages training checkpoints including save, load, resume, sync to Hugging Face,
and cleanup of old checkpoints.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus.core.exceptions import JobStateError
from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


class CheckpointManager:
    """Manages training checkpoints."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.config = config or {}
        
        # Checkpoint settings
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "./workspace/checkpoints"))
        self.max_checkpoints = self.config.get("max_checkpoints", 5)
        self.max_checkpoint_size_mb = self.config.get("max_checkpoint_size_mb", 10000)
        self.cleanup_old_checkpoints = self.config.get("cleanup_old_checkpoints", True)
        self.save_checkpoints = self.config.get("save_checkpoints", True)
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        job_id: str,
        epoch: int,
        step: int,
        loss: Optional[float],
        model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save a training checkpoint."""
        if not self.save_checkpoints:
            return {"saved": False, "reason": "checkpoints disabled"}
        
        checkpoint_name = f"{job_id}-epoch{epoch}-step{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint metadata
        checkpoint_meta = {
            "job_id": job_id,
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "saved_at": time.time(),
            "model_path": model_path,
            "adapter_path": adapter_path,
            "metadata": metadata or {},
        }
        
        meta_file = checkpoint_path / "checkpoint_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(checkpoint_meta, f, indent=2)
        
        # Copy model files if provided
        if model_path and Path(model_path).exists():
            dest_model = checkpoint_path / "model"
            if dest_model.exists():
                shutil.rmtree(dest_model)
            shutil.copytree(model_path, dest_model)
            checkpoint_meta["model_copied"] = True
        
        # Copy adapter files if provided
        if adapter_path and Path(adapter_path).exists():
            dest_adapter = checkpoint_path / "adapter"
            if dest_adapter.exists():
                shutil.rmtree(dest_adapter)
            shutil.copytree(adapter_path, dest_adapter)
            checkpoint_meta["adapter_copied"] = True
        
        # Update metadata file
        with open(meta_file, 'w') as f:
            json.dump(checkpoint_meta, f, indent=2)
        
        self.logger.info(
            f"Checkpoint saved: {checkpoint_name} (epoch={epoch}, step={step}, loss={loss})"
        )
        
        # Cleanup old checkpoints if needed
        if self.cleanup_old_checkpoints:
            self._cleanup_old_checkpoints(job_id)
        
        return {
            "saved": True,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_name": checkpoint_name,
            "epoch": epoch,
            "step": step,
            "loss": loss,
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a training checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            raise JobStateError(f"Checkpoint not found: {checkpoint_path}")
        
        meta_file = checkpoint_dir / "checkpoint_meta.json"
        
        if not meta_file.exists():
            raise JobStateError(f"Checkpoint metadata not found: {meta_file}")
        
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        self.logger.info(
            f"Checkpoint loaded: {checkpoint_path} (epoch={metadata.get('epoch')}, "
            f"step={metadata.get('step')})"
        )
        
        return metadata
    
    def list_checkpoints(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all checkpoints, optionally filtered by job ID."""
        checkpoints = []
        
        if not self.checkpoint_dir.exists():
            return checkpoints
        
        for checkpoint_dir in sorted(self.checkpoint_dir.iterdir()):
            if not checkpoint_dir.is_dir():
                continue
            
            meta_file = checkpoint_dir / "checkpoint_meta.json"
            if not meta_file.exists():
                continue
            
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Filter by job ID if specified
            if job_id and metadata.get("job_id") != job_id:
                continue
            
            # Calculate checkpoint size
            size_bytes = sum(
                f.stat().st_size for f in checkpoint_dir.rglob("*") if f.is_file()
            )
            
            checkpoints.append({
                "path": str(checkpoint_dir),
                "name": checkpoint_dir.name,
                "job_id": metadata.get("job_id"),
                "epoch": metadata.get("epoch"),
                "step": metadata.get("step"),
                "loss": metadata.get("loss"),
                "saved_at": metadata.get("saved_at"),
                "size_bytes": size_bytes,
                "size_mb": size_bytes / (1024 ** 2),
            })
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """Delete a checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            return False
        
        try:
            shutil.rmtree(checkpoint_dir)
            self.logger.info(f"Checkpoint deleted: {checkpoint_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint: {str(e)}")
            return False
    
    def get_latest_checkpoint(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint for a job."""
        checkpoints = self.list_checkpoints(job_id)
        
        if not checkpoints:
            return None
        
        # Sort by saved_at descending
        checkpoints.sort(key=lambda c: c.get("saved_at", 0), reverse=True)
        return checkpoints[0]
    
    def _cleanup_old_checkpoints(self, job_id: str) -> None:
        """Clean up old checkpoints for a job, keeping only the most recent."""
        checkpoints = self.list_checkpoints(job_id)
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by saved_at ascending (oldest first)
        checkpoints.sort(key=lambda c: c.get("saved_at", 0))
        
        # Delete oldest checkpoints
        to_delete = len(checkpoints) - self.max_checkpoints
        for checkpoint in checkpoints[:to_delete]:
            self.delete_checkpoint(checkpoint["path"])
            self.logger.info(
                f"Cleaned up old checkpoint: {checkpoint['name']}"
            )
    
    def get_checkpoint_size_mb(self, checkpoint_path: str) -> float:
        """Get checkpoint size in MB."""
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            return 0.0
        
        size_bytes = sum(
            f.stat().st_size for f in checkpoint_dir.rglob("*") if f.is_file()
        )
        
        return size_bytes / (1024 ** 2)
    
    def sync_to_huggingface(
        self,
        checkpoint_path: str,
        repo_id: str,
        hf_token: str,
    ) -> Dict[str, Any]:
        """Sync a checkpoint to Hugging Face Hub."""
        from huggingface_hub import HfApi, create_repo, upload_folder
        
        self.nexus_logger.log_event(
            "checkpoint_sync_start",
            {
                "checkpoint_path": checkpoint_path,
                "repo_id": repo_id,
            }
        )
        
        try:
            api = HfApi(token=hf_token)
            
            # Create repo if it doesn't exist
            try:
                create_repo(repo_id, token=hf_token, exist_ok=True, private=True)
            except Exception as e:
                self.logger.warning(f"Repo creation warning: {str(e)}")
            
            # Upload checkpoint folder
            upload_folder(
                folder_path=checkpoint_path,
                repo_id=repo_id,
                token=hf_token,
                repo_type="model",
            )
            
            self.nexus_logger.log_event(
                "checkpoint_sync_completed",
                {
                    "checkpoint_path": checkpoint_path,
                    "repo_id": repo_id,
                }
            )
            
            return {
                "synced": True,
                "repo_id": repo_id,
                "checkpoint_path": checkpoint_path,
                "synced_at": time.time(),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to sync checkpoint to HF: {str(e)}")
            
            self.nexus_logger.log_event(
                "checkpoint_sync_failed",
                {
                    "checkpoint_path": checkpoint_path,
                    "repo_id": repo_id,
                    "error": str(e),
                }
            )
            
            return {
                "synced": False,
                "repo_id": repo_id,
                "error": str(e),
            }
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get overall checkpoint statistics."""
        checkpoints = self.list_checkpoints()
        
        total_size_mb = sum(c.get("size_mb", 0) for c in checkpoints)
        total_count = len(checkpoints)
        
        jobs = set(c.get("job_id") for c in checkpoints if c.get("job_id"))
        
        return {
            "total_checkpoints": total_count,
            "total_size_mb": total_size_mb,
            "total_size_gb": total_size_mb / 1024,
            "jobs_with_checkpoints": len(jobs),
            "checkpoint_dir": str(self.checkpoint_dir),
        }