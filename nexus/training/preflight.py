"""
Preflight checks for NEXUS training.

Validates that a model can be fine-tuned on the selected GPU before
submitting a training job. Checks memory requirements, quantization
compatibility, and resource constraints.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus.core.exceptions import ProviderError, ValidationError
from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


@dataclass
class PreflightResult:
    """Result of a preflight check."""
    
    feasible: bool
    gpu_type: str
    gpu_memory_gb: float
    estimated_memory_gb: float
    memory_headroom_gb: float
    recommended_method: str
    batch_size: int
    gradient_accumulation_steps: int
    use_gradient_checkpointing: bool
    use_mixed_precision: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible": self.feasible,
            "gpu_type": self.gpu_type,
            "gpu_memory_gb": self.gpu_memory_gb,
            "estimated_memory_gb": self.estimated_memory_gb,
            "memory_headroom_gb": self.memory_headroom_gb,
            "recommended_method": self.recommended_method,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_mixed_precision": self.use_mixed_precision,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class PreflightChecker:
    """Checks training feasibility on target hardware."""
    
    # Memory overhead factors (multipliers on model size)
    MEMORY_FACTORS = {
        "sft": 4.0,      # Full fine-tuning: model + grads + optimizer states + activations
        "lora": 1.5,     # LoRA: model + small adapter + activations
        "qlora": 0.8,    # QLoRA: 4-bit quantized model + adapter + activations
    }
    
    # GPU specifications
    GPU_SPECS = {
        "Tesla T4": {"memory_gb": 14.56, "supports_bfloat16": True, "supports_fp16": True},
        "Tesla V100": {"memory_gb": 16.0, "supports_bfloat16": False, "supports_fp16": True},
        "Tesla A100": {"memory_gb": 40.0, "supports_bfloat16": True, "supports_fp16": True},
        "Tesla A10G": {"memory_gb": 24.0, "supports_bfloat16": True, "supports_fp16": True},
        "RTX 3090": {"memory_gb": 24.0, "supports_bfloat16": True, "supports_fp16": True},
        "RTX 4090": {"memory_gb": 24.0, "supports_bfloat16": True, "supports_fp16": True},
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.config = config or {}
        self.max_memory_usage = self.config.get("max_memory_usage", 0.8)
        self.default_batch_size = self.config.get("default_batch_size", 2)
        self.default_gradient_accumulation = self.config.get(
            "default_gradient_accumulation", 4
        )
    
    def check_feasibility(
        self,
        model_parameter_count: int,
        gpu_type: str = "Tesla T4",
        method: str = "qlora",
        batch_size: Optional[int] = None,
        max_sequence_length: int = 2048,
    ) -> PreflightResult:
        """Check if training is feasible on the specified GPU."""
        self.nexus_logger.log_event(
            "preflight_check_start",
            {
                "model_parameter_count": model_parameter_count,
                "gpu_type": gpu_type,
                "method": method,
                "batch_size": batch_size,
                "max_sequence_length": max_sequence_length,
            }
        )
        
        warnings: List[str] = []
        errors: List[str] = []
        
        # Get GPU specs
        gpu_spec = self.GPU_SPECS.get(
            gpu_type, {"memory_gb": 14.56, "supports_bfloat16": True, "supports_fp16": True}
        )
        gpu_memory_gb = gpu_spec["memory_gb"]
        
        # Determine training method
        if method not in ("sft", "lora", "qlora"):
            method = "qlora"
            warnings.append(f"Unknown method '{method}', defaulting to qlora")
        
        # Calculate estimated memory per method (in GB)
        # Bytes per parameter: fp16/bf16=2, 4-bit=0.5
        if method == "qlora":
            # Quantized weights (4-bit) + small LoRA adapters + CUDA overhead
            weights_gb = (model_parameter_count * 0.5) / (1024 ** 3)
            adapters_gb = (model_parameter_count * 0.02) / (1024 ** 3)
            cuda_overhead_gb = 1.0
        elif method == "lora":
            # Full weights (fp16) + gradients + small LoRA adapters + CUDA overhead
            weights_gb = (model_parameter_count * 2) / (1024 ** 3)
            gradients_gb = (model_parameter_count * 2) / (1024 ** 3)
            adapters_gb = (model_parameter_count * 0.02) / (1024 ** 3)
            cuda_overhead_gb = 1.0
            weights_gb += gradients_gb
        else:  # sft
            # Full weights + gradients + Adam optimizer states + CUDA overhead
            weights_gb = (model_parameter_count * 2) / (1024 ** 3)
            gradients_gb = (model_parameter_count * 2) / (1024 ** 3)
            optimizer_states_gb = (model_parameter_count * 8) / (1024 ** 3)
            adapters_gb = 0.0
            cuda_overhead_gb = 1.0
            weights_gb += gradients_gb + optimizer_states_gb
        
        # Add activation memory based on sequence length and batch size
        # Empirical estimate: activations ≈ batch_size * seq_len * 0.0002 GB
        effective_batch_size = batch_size or self.default_batch_size
        activation_memory_gb = (
            effective_batch_size * max_sequence_length * 0.0002
        )  # Rough estimate in GB
        
        estimated_memory_gb = weights_gb + adapters_gb + cuda_overhead_gb + activation_memory_gb
        
        # Apply memory headroom (leave 20% for system overhead)
        usable_memory_gb = gpu_memory_gb * self.max_memory_usage
        memory_headroom_gb = usable_memory_gb - estimated_memory_gb
        
        # Determine feasibility
        feasible = estimated_memory_gb <= usable_memory_gb
        
        if not feasible:
            errors.append(
                f"Estimated memory {estimated_memory_gb:.2f}GB exceeds "
                f"usable GPU memory {usable_memory_gb:.2f}GB on {gpu_type}"
            )
        
        # Check method compatibility
        if method == "sft" and gpu_memory_gb < 24:
            warnings.append(
                f"SFT fine-tuning on {gpu_type} ({gpu_memory_gb}GB) may be tight. "
                "Consider LoRA or QLoRA instead."
            )
        
        if method == "qlora" and not gpu_spec["supports_bfloat16"]:
            warnings.append(
                f"{gpu_type} does not support bfloat16. QLoRA will use fp16 instead."
            )
        
        # Determine gradient accumulation steps
        gradient_accumulation_steps = self.default_gradient_accumulation
        if estimated_memory_gb > usable_memory_gb * 0.7:
            gradient_accumulation_steps = max(
                gradient_accumulation_steps * 2, 8
            )
            warnings.append(
                f"Increasing gradient accumulation to {gradient_accumulation_steps} "
                "to reduce memory usage"
            )
        
        # Determine mixed precision
        use_mixed_precision = gpu_spec["supports_bfloat16"]
        if not use_mixed_precision:
            use_mixed_precision = gpu_spec["supports_fp16"]
            if not use_mixed_precision:
                warnings.append("No mixed precision support detected; training may be slow")
        
        use_gradient_checkpointing = estimated_memory_gb > usable_memory_gb * 0.6
        
        # Recommended method
        if not feasible:
            if method != "qlora":
                # Try qlora as fallback (most memory-efficient)
                fallback_result = self.check_feasibility(
                    model_parameter_count=model_parameter_count,
                    gpu_type=gpu_type,
                    method="qlora",
                    batch_size=effective_batch_size,
                    max_sequence_length=max_sequence_length,
                )
                if fallback_result.feasible:
                    recommended_method = "qlora"
                    warnings.append(
                        f"{method} not feasible, but qlora may work "
                        f"({fallback_result.estimated_memory_gb:.2f}GB estimated)"
                    )
                    # Use qlora result values
                    estimated_memory_gb = fallback_result.estimated_memory_gb
                    memory_headroom_gb = fallback_result.memory_headroom_gb
                else:
                    recommended_method = "none"
                    errors.append(
                        f"No training method feasible for {model_parameter_count:,} parameter "
                        f"model on {gpu_type} ({gpu_memory_gb}GB)"
                    )
            else:
                recommended_method = "none"
                errors.append(
                    f"No training method feasible for {model_parameter_count:,} parameter "
                    f"model on {gpu_type} ({gpu_memory_gb}GB)"
                )
        elif method == "sft" and gpu_memory_gb < 24:
            recommended_method = "qlora"
            warnings.append("Switched to QLoRA due to limited GPU memory")
        else:
            recommended_method = method
        
        result = PreflightResult(
            feasible=feasible and recommended_method != "none",
            gpu_type=gpu_type,
            gpu_memory_gb=gpu_memory_gb,
            estimated_memory_gb=estimated_memory_gb,
            memory_headroom_gb=memory_headroom_gb,
            recommended_method=recommended_method,
            batch_size=effective_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_mixed_precision=use_mixed_precision,
            warnings=warnings,
            errors=errors,
        )
        
        self.nexus_logger.log_event(
            "preflight_check_completed",
            {
                "feasible": result.feasible,
                "recommended_method": result.recommended_method,
                "estimated_memory_gb": result.estimated_memory_gb,
                "errors": result.errors,
                "warnings": result.warnings,
            }
        )
        
        return result
    
    def check_model_compatibility(
        self,
        model_architecture: str,
        method: str = "qlora",
    ) -> Dict[str, Any]:
        """Check if a model architecture is compatible with a training method."""
        result = {
            "compatible": True,
            "warnings": [],
            "errors": [],
        }
        
        arch_lower = model_architecture.lower()
        
        # Check for known incompatible architectures
        if "encoder_only" in arch_lower and method in ("lora", "qlora"):
            result["warnings"].append(
                f"Encoder-only architecture ({model_architecture}) may not benefit from LoRA"
            )
        
        if "t5" in arch_lower and method == "qlora":
            result["warnings"].append(
                "QLoRA on T5 models is experimental; use with caution"
            )
        
        # Check for MoE (Mixture of Experts)
        if "moe" in arch_lower or "mixtral" in arch_lower:
            if method == "qlora":
                result["warnings"].append(
                    "QLoRA on MoE models may have uneven memory distribution across experts"
                )
        
        return result
    
    def get_max_feasible_parameters(
        self,
        gpu_type: str = "Tesla T4",
        method: str = "qlora",
        batch_size: int = 2,
        max_sequence_length: int = 2048,
    ) -> int:
        """Get the maximum feasible parameter count for a GPU/method combination."""
        gpu_spec = self.GPU_SPECS.get(
            gpu_type, {"memory_gb": 14.56, "supports_bfloat16": True, "supports_fp16": True}
        )
        gpu_memory_gb = gpu_spec["memory_gb"]
        usable_memory_gb = gpu_memory_gb * self.max_memory_usage
        
        # Bytes per parameter for weights
        if method == "qlora":
            bytes_per_param = 0.5  # 4-bit quantized
            adapter_bytes_per_param = 0.02  # LoRA adapters
        elif method == "lora":
            bytes_per_param = 4.0  # weights + gradients (fp16)
            adapter_bytes_per_param = 0.02
        else:  # sft
            bytes_per_param = 12.0  # weights + gradients + Adam states
            adapter_bytes_per_param = 0.0
        
        cuda_overhead_gb = 1.0
        activation_gb = batch_size * max_sequence_length * 0.0002
        
        # Solve: params * (bytes_per_param + adapter_bytes) / 1024^3 + cuda + activation <= usable
        bytes_per_param_total = bytes_per_param + adapter_bytes_per_param
        
        available_for_params = usable_memory_gb - cuda_overhead_gb - activation_gb
        
        if available_for_params <= 0:
            return 0
        
        max_params = int(
            (available_for_params * (1024 ** 3)) / bytes_per_param_total
        )
        
        return max_params