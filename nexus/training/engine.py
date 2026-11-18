"""
Training engine for NEXUS.

Provides functionality to train and fine-tune models using various training
methods like LoRA, QLoRA, and supervised fine-tuning.
"""

from typing import Dict, List, Optional, Any

from nexus.core.config import Config
from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


class TrainingEngine:
    """Engine for training and fine-tuning models."""
    
    def __init__(self, config: Config):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.config = config
        
        # Training configuration
        training_config = config.get("training")
        if hasattr(training_config, "method"):
            self.default_method = training_config.method
        else:
            self.default_method = "qlora"
            
        if hasattr(training_config, "batch_size"):
            self.default_batch_size = training_config.batch_size
        else:
            self.default_batch_size = 2
            
        if hasattr(training_config, "learning_rate"):
            self.default_learning_rate = training_config.learning_rate
        else:
            self.default_learning_rate = 0.0001
            
        if hasattr(training_config, "epochs"):
            self.default_epochs = training_config.epochs
        else:
            self.default_epochs = 3
            
        if hasattr(training_config, "max_sequence_length"):
            self.default_max_sequence_length = training_config.max_sequence_length
        else:
            self.default_max_sequence_length = 2048
    
    def create_training_job(self, job_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Create a training job."""
        self.nexus_logger.log_event(
            "training_job_created",
            {
                "job_spec": job_spec
            }
        )
        
        # Create training job
        training_job = {
            "job_id": job_spec.get("job_id", f"training_{int(time.time())}"),
            "model": job_spec.get("model", "Qwen2.5-0.5B-Instruct"),
            "dataset": job_spec.get("dataset", "python-code-instructions"),
            "method": job_spec.get("method", self.default_method),
            "batch_size": job_spec.get("batch_size", self.default_batch_size),
            "learning_rate": job_spec.get("learning_rate", self.default_learning_rate),
            "epochs": job_spec.get("epochs", self.default_epochs),
            "max_sequence_length": job_spec.get("max_sequence_length", self.default_max_sequence_length),
            "status": "pending",
            "progress": 0.0,
            "created_at": time.time(),
            "estimated_time_seconds": self._estimate_training_time(job_spec),
            "estimated_gpu_memory_gb": self._estimate_gpu_memory(job_spec)
        }
        
        self.nexus_logger.log_event(
            "training_job_created",
            {
                "job_id": training_job["job_id"],
                "model": training_job["model"],
                "dataset": training_job["dataset"],
                "method": training_job["method"],
                "estimated_time_seconds": training_job["estimated_time_seconds"]
            }
        )
        
        return training_job
    
    def _estimate_training_time(self, job_spec: Dict[str, Any]) -> float:
        """Estimate training time based on job specification."""
        # Simple estimation based on dataset size and model size
        dataset_size = job_spec.get("dataset_size", 1000000)
        model_params = job_spec.get("model_parameters", 500000000)
        
        # Rough estimate: 0.1 seconds per sample per parameter (simplified)
        estimated_seconds = (dataset_size * model_params) / 1e10
        
        # Cap at reasonable values
        return min(estimated_seconds, 7200.0)  # Max 2 hours
    
    def _estimate_gpu_memory(self, job_spec: Dict[str, Any]) -> float:
        """Estimate GPU memory requirements."""
        # Based on model size and batch size
        model_params = job_spec.get("model_parameters", 500000000)
        batch_size = job_spec.get("batch_size", self.default_batch_size)
        
        # Rough estimate: 8 bytes per parameter * model params * batch_size / 1024^3
        estimated_gb = (model_params * 8 * batch_size) / (1024 ** 3)
        
        # Cap at reasonable values
        return min(estimated_gb, 24.0)  # Max 24 GB
    
    def get_training_methods(self) -> List[str]:
        """Get available training methods."""
        return ["sft", "lora", "qlora", "peft"]
    
    def get_recommended_settings(self, model_size: str) -> Dict[str, Any]:
        """Get recommended training settings for a model size."""
        settings_map = {
            "small": {
                "method": "lora",
                "batch_size": 2,
                "learning_rate": 0.0001,
                "epochs": 3,
                "max_sequence_length": 1024
            },
            "medium": {
                "method": "qlora",
                "batch_size": 4,
                "learning_rate": 0.0001,
                "epochs": 2,
                "max_sequence_length": 2048
            },
            "large": {
                "method": "qlora",
                "batch_size": 8,
                "learning_rate": 0.0001,
                "epochs": 2,
                "max_sequence_length": 2048
            }
        }
        
        # Determine model size category
        if "0.5B" in model_size:
            size_category = "small"
        elif "7B" in model_size or "13B" in model_size:
            size_category = "medium"
        else:
            size_category = "large"
        
        return settings_map.get(size_category, settings_map["medium"])


# Import here to avoid circular dependencies
import time