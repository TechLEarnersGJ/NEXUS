"""
QLoRA (Quantized LoRA) training for NEXUS.

Generates QLoRA training configurations with 4-bit quantization for
memory-efficient fine-tuning of large language models on consumer GPUs.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


@dataclass
class QLoRAConfig:
    """QLoRA training configuration."""
    
    # QLoRA quantization settings
    bits: int = 4
    compute_dtype: str = "bfloat16"
    quant_type: str = "nf4"
    double_quant: bool = True
    
    # LoRA hyperparameters
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # Training hyperparameters
    learning_rate: float = 0.0001
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_sequence_length: int = 2048
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    
    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: str = "bfloat16"
    
    # Output
    output_dir: str = "./outputs"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    logging_steps: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bits": self.bits,
            "compute_dtype": self.compute_dtype,
            "quant_type": self.quant_type,
            "double_quant": self.double_quant,
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_epochs": self.num_epochs,
            "max_sequence_length": self.max_sequence_length,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "gradient_checkpointing": self.gradient_checkpointing,
            "mixed_precision": self.mixed_precision,
            "output_dir": self.output_dir,
            "save_strategy": self.save_strategy,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
        }


class QLoRATrainer:
    """Generates QLoRA training jobs and configurations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.config = config or {}
        
        # Default QLoRA settings
        self.default_r = self.config.get("default_r", 16)
        self.default_lora_alpha = self.config.get("default_lora_alpha", 32)
        self.default_learning_rate = self.config.get("default_learning_rate", 0.0001)
        self.default_batch_size = self.config.get("default_batch_size", 2)
        self.default_epochs = self.config.get("default_epochs", 3)
        self.default_max_sequence_length = self.config.get("default_max_sequence_length", 2048)
        self.default_use_4bit = self.config.get("default_use_4bit", True)
        self.default_bnb_4bit_compute_dtype = self.config.get(
            "default_bnb_4bit_compute_dtype", "bfloat16"
        )
        self.default_bnb_4bit_quant_type = self.config.get(
            "default_bnb_4bit_quant_type", "nf4"
        )
        self.default_bnb_4bit_double_quant = self.config.get(
            "default_bnb_4bit_double_quant", True
        )
    
    def create_config(
        self,
        model_id: str,
        dataset_id: str,
        method_overrides: Optional[Dict[str, Any]] = None,
    ) -> QLoRAConfig:
        """Create a QLoRA training configuration."""
        self.nexus_logger.log_event(
            "qlora_config_created",
            {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "overrides": method_overrides or {},
            }
        )
        
        config = QLoRAConfig(
            r=self.default_r,
            lora_alpha=self.default_lora_alpha,
            learning_rate=self.default_learning_rate,
            batch_size=self.default_batch_size,
            num_epochs=self.default_epochs,
            max_sequence_length=self.default_max_sequence_length,
            bits=4 if self.default_use_4bit else 8,
            compute_dtype=self.default_bnb_4bit_compute_dtype,
            quant_type=self.default_bnb_4bit_quant_type,
            double_quant=self.default_bnb_4bit_double_quant,
        )
        
        # Apply overrides
        if method_overrides:
            for key, value in method_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.nexus_logger.log_event(
            "qlora_config_completed",
            {
                "config": config.to_dict(),
            }
        )
        
        return config
    
    def create_job_spec(
        self,
        job_id: str,
        model_id: str,
        dataset_id: str,
        config: Optional[QLoRAConfig] = None,
        provider: str = "kaggle",
        preflight_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a complete QLoRA training job specification."""
        if config is None:
            config = self.create_config(model_id, dataset_id)
        
        # Apply preflight adjustments
        if preflight_result:
            config.batch_size = preflight_result.get("batch_size", config.batch_size)
            config.gradient_accumulation_steps = preflight_result.get(
                "gradient_accumulation_steps", config.gradient_accumulation_steps
            )
            config.gradient_checkpointing = preflight_result.get(
                "use_gradient_checkpointing", config.gradient_checkpointing
            )
            config.mixed_precision = preflight_result.get(
                "use_mixed_precision", config.mixed_precision
            )
        
        job_spec = {
            "job_id": job_id,
            "name": f"QLoRA Training: {model_id}",
            "provider": provider,
            "method": "qlora",
            "description": f"QLoRA fine-tuning of {model_id} on {dataset_id}",
            "config": {
                "base_model_id": model_id,
                "dataset_id": dataset_id,
                "qlora_config": config.to_dict(),
                "training_method": "qlora",
                "output_dir": config.output_dir,
            },
            "requirements": {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "method": "qlora",
                "bits": config.bits,
            },
            "resources": {
                "gpu_memory_gb": preflight_result.get("gpu_memory_gb", 14.56) if preflight_result else 14.56,
                "estimated_memory_gb": preflight_result.get("estimated_memory_gb", 6.0) if preflight_result else 6.0,
            },
            "metadata": {
                "created_at": time.time(),
                "model_id": model_id,
                "dataset_id": dataset_id,
                "method": "qlora",
            },
            "priority": 1,
            "timeout_seconds": 86400,  # 24 hours
            "max_retries": 3,
            "retry_delay": 60.0,
        }
        
        self.nexus_logger.log_event(
            "qlora_job_spec_created",
            {
                "job_id": job_id,
                "model_id": model_id,
                "dataset_id": dataset_id,
                "provider": provider,
                "config": config.to_dict(),
            }
        )
        
        return job_spec
    
    def get_recommended_settings(
        self,
        model_parameter_count: int,
        gpu_memory_gb: float = 14.56,
    ) -> Dict[str, Any]:
        """Get recommended QLoRA settings for a model/GPU combination."""
        # Scale r and alpha based on model size
        if model_parameter_count <= 1_000_000_000:  # <= 1B
            r = 16
            lora_alpha = 32
            batch_size = 4
        elif model_parameter_count <= 7_000_000_000:  # <= 7B
            r = 16
            lora_alpha = 32
            batch_size = 2
        elif model_parameter_count <= 13_000_000_000:  # <= 13B
            r = 8
            lora_alpha = 16
            batch_size = 1
        else:  # > 13B
            r = 8
            lora_alpha = 16
            batch_size = 1
        
        # Adjust for GPU memory
        if gpu_memory_gb < 16:
            batch_size = max(1, batch_size // 2)
            gradient_accumulation_steps = 8
        else:
            gradient_accumulation_steps = 4
        
        return {
            "bits": 4,
            "compute_dtype": "bfloat16",
            "quant_type": "nf4",
            "double_quant": True,
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0.05,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": 0.0001,
            "num_epochs": 3,
            "max_sequence_length": 2048,
            "gradient_checkpointing": True,
            "mixed_precision": "bfloat16",
        }