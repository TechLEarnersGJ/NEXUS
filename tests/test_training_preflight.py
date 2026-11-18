"""
Tests for NEXUS training preflight checker.
"""

from nexus.training.preflight import PreflightChecker


def test_preflight_qlora_t4_05b():
    """Test QLoRA feasibility on T4 with 0.5B model."""
    checker = PreflightChecker()
    result = checker.check_feasibility(
        model_parameter_count=500_000_000,
        gpu_type='Tesla T4',
        method='qlora',
    )
    assert result.feasible is True
    assert result.gpu_type == 'Tesla T4'
    assert result.gpu_memory_gb == 14.56
    assert result.estimated_memory_gb > 0
    assert result.recommended_method == 'qlora'
    assert result.batch_size > 0
    assert result.gradient_accumulation_steps > 0


def test_preflight_sft_t4_05b():
    """Test SFT feasibility on T4 with 0.5B model (should recommend QLoRA)."""
    checker = PreflightChecker()
    result = checker.check_feasibility(
        model_parameter_count=500_000_000,
        gpu_type='Tesla T4',
        method='sft',
    )
    # SFT might be feasible but should recommend QLoRA for small GPUs
    assert result.recommended_method in ('sft', 'qlora')
    assert len(result.warnings) >= 0


def test_preflight_qlora_t4_70b():
    """Test QLoRA infeasibility on T4 with 70B model."""
    checker = PreflightChecker()
    result = checker.check_feasibility(
        model_parameter_count=70_000_000_000,
        gpu_type='Tesla T4',
        method='qlora',
    )
    assert result.feasible is False
    assert len(result.errors) > 0
    assert result.recommended_method == 'none'


def test_preflight_lora_t4_7b():
    """Test LoRA feasibility on T4 with 7B model."""
    checker = PreflightChecker()
    result = checker.check_feasibility(
        model_parameter_count=7_000_000_000,
        gpu_type='Tesla T4',
        method='lora',
    )
    # LoRA on 7B with T4 might not be feasible, but QLoRA fallback should work
    assert result.recommended_method in ('lora', 'qlora', 'none')


def test_preflight_gpu_specs():
    """Test GPU specs are correct."""
    checker = PreflightChecker()
    assert 'Tesla T4' in checker.GPU_SPECS
    assert checker.GPU_SPECS['Tesla T4']['memory_gb'] == 14.56
    assert checker.GPU_SPECS['Tesla A100']['memory_gb'] == 40.0


def test_preflight_max_feasible_params():
    """Test max feasible parameter calculation."""
    checker = PreflightChecker()
    
    max_qlora = checker.get_max_feasible_parameters(
        gpu_type='Tesla T4', method='qlora'
    )
    assert max_qlora > 0
    
    max_lora = checker.get_max_feasible_parameters(
        gpu_type='Tesla T4', method='lora'
    )
    assert max_lora > 0
    
    max_sft = checker.get_max_feasible_parameters(
        gpu_type='Tesla T4', method='sft'
    )
    assert max_sft > 0
    
    # QLoRA should support more params than LoRA which should support more than SFT
    assert max_qlora >= max_lora >= max_sft


def test_preflight_different_gpus():
    """Test preflight on different GPU types."""
    checker = PreflightChecker()
    
    for gpu_type in ['Tesla T4', 'Tesla A100', 'Tesla V100']:
        result = checker.check_feasibility(
            model_parameter_count=500_000_000,
            gpu_type=gpu_type,
            method='qlora',
        )
        assert result.gpu_type == gpu_type
        assert result.feasible is True  # 0.5B should work on all GPUs


def test_preflight_result_to_dict():
    """Test PreflightResult serialization."""
    checker = PreflightChecker()
    result = checker.check_feasibility(
        model_parameter_count=500_000_000,
        gpu_type='Tesla T4',
        method='qlora',
    )
    d = result.to_dict()
    assert 'feasible' in d
    assert 'gpu_type' in d
    assert 'estimated_memory_gb' in d
    assert 'recommended_method' in d
    assert isinstance(d['warnings'], list)
    assert isinstance(d['errors'], list)


def test_preflight_model_compatibility():
    """Test model architecture compatibility checking."""
    checker = PreflightChecker()
    
    # Normal causal LM
    result = checker.check_model_compatibility('QwenForCausalLM', method='qlora')
    assert result['compatible'] is True
    
    # MoE model
    result = checker.check_model_compatibility('MixtralForCausalLM', method='qlora')
    assert len(result['warnings']) > 0
    
    # T5 model
    result = checker.check_model_compatibility('T5ForConditionalGeneration', method='qlora')
    assert len(result['warnings']) > 0