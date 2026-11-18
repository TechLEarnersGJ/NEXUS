"""
Tests for NEXUS provider abstraction and fake provider lifecycle.
"""

import os
import tempfile
import time
from pathlib import Path

from nexus.providers.manager import ProviderManager
from nexus.providers.base import (
    ComputeProvider,
    Capabilities,
    Availability,
    QuotaInfo,
    JobSpec,
    JobHandle,
    JobStatus,
    AuthResult,
    OutputResult,
    AuthMethod,
)
from nexus.providers.fake import (
    FakeKaggleProvider,
    FakeColabProvider,
    FakeHuggingFaceProvider,
    FakeProvider,
)
from nexus.core.config import Config
from nexus.core.exceptions import (
    AuthError,
    ProviderError,
    ProviderUnavailable,
    QuotaExceeded,
    JobStateError,
    ValidationError,
)


def test_fake_kaggle_full_lifecycle():
    """Test complete fake Kaggle provider lifecycle."""
    config = {
        "timeout_seconds": 30,
        "max_retries": 3,
        "retry_delay": 1.0,
        "auth_env_vars": {},
    }
    
    provider = FakeKaggleProvider(config)
    
    # 1. Authenticate
    auth_result = provider.authenticate()
    assert auth_result.authenticated is True
    assert auth_result.provider == "fake_kaggle"
    assert auth_result.token is not None
    
    # 2. Check capabilities
    caps = provider.capabilities()
    assert caps.gpu_memory_gb > 0
    assert caps.resume is True
    assert caps.supports_notebooks is True
    
    # 3. Check availability
    availability = provider.check_availability()
    assert availability.available is True
    assert availability.gpu_available is True
    
    # 4. Check quota
    quota = provider.quota()
    assert quota is not None
    assert quota.remaining > 0
    assert quota.used < quota.limit  # Quota not exceeded
    
    # 5. Submit job
    job_spec = JobSpec(
        job_id="test_job_123",
        name="Test Training Job",
        provider="fake_kaggle",
        config={"base_model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
        priority=1,
    )
    
    handle = provider.submit_job(job_spec)
    assert handle.job_id == "test_job_123"
    assert handle.provider == "fake_kaggle"
    assert handle.handle is not None
    
    # 6. Get status - wait for completion
    import time
    status = provider.get_status(handle)
    # Wait for job to complete
    for _ in range(20):
        if status.state == "completed":
            break
        time.sleep(0.1)
        status = provider.get_status(handle)
    
    assert status.state == "completed"
    
    # 7. Retrieve output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_result = provider.retrieve_output(handle, tmpdir)
        assert output_result.job_id == "test_job_123"
        assert output_result.output_path is not None
        assert len(output_result.output_files) > 0
    
    # 8. Cancel job
    handle2 = provider.submit_job(JobSpec(
        job_id="test_job_cancel",
        name="Cancel Test",
        provider="fake_kaggle",
        config={},
    ))
    
    # Cancel before completion
    try:
        provider.cancel_job(handle2)
        # Cancelled successfully
    except JobStateError:
        # May have already completed - acceptable
        pass


def test_fake_colab_full_lifecycle():
    """Test complete fake Colab provider lifecycle."""
    config = {
        "timeout_seconds": 30,
        "max_retries": 3,
        "retry_delay": 1.0,
        "auth_env_vars": {},
    }
    
    provider = FakeColabProvider(config)
    
    # 1. Authenticate
    auth_result = provider.authenticate()
    assert auth_result.authenticated is True
    assert auth_result.method == AuthMethod.OAUTH
    
    # 2. Check capabilities
    caps = provider.capabilities()
    assert caps.supports_colab is True
    assert caps.resume is False  # Colab sessions are ephemeral
    
    # 3. Check availability
    availability = provider.check_availability()
    assert availability.available is True
    
    # 4. Submit job
    job_spec = JobSpec(
        job_id="colab_job_123",
        name="Colab Training",
        provider="fake_colab",
        config={"base_model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
    )
    
    handle = provider.submit_job(job_spec)
    assert handle.job_id == "colab_job_123"
    
    # 5. Get status - wait for completion
    import time
    status = provider.get_status(handle)
    for _ in range(20):
        if status.state == "completed":
            break
        time.sleep(0.1)
        status = provider.get_status(handle)
    
    assert status.state == "completed"
    
    # 6. Retrieve output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_result = provider.retrieve_output(handle, tmpdir)
        assert output_result.job_id == "colab_job_123"
        assert "model" in output_result.metadata


def test_fake_huggingface_full_lifecycle():
    """Test complete fake Hugging Face provider lifecycle."""
    config = {
        "timeout_seconds": 30,
        "max_retries": 3,
        "retry_delay": 1.0,
        "auth_env_vars": {},
    }
    
    provider = FakeHuggingFaceProvider(config)
    
    # 1. Authenticate
    auth_result = provider.authenticate()
    assert auth_result.authenticated is True
    assert auth_result.method == AuthMethod.TOKEN
    
    # 2. Check capabilities
    caps = provider.capabilities()
    assert caps.supports_huggingface is True
    assert caps.gpu_count == 0  # No GPU on HF side
    
    # 3. Check availability
    availability = provider.check_availability()
    assert availability.available is True
    assert availability.gpu_available is False
    
    # 4. Submit job
    job_spec = JobSpec(
        job_id="hf_job_123",
        name="HF Model Upload",
        provider="fake_huggingface",
        config={"operation": "model_upload"},
    )
    
    handle = provider.submit_job(job_spec)
    assert handle.job_id == "hf_job_123"
    
    # 5. Get status - wait for completion
    import time
    status = provider.get_status(handle)
    for _ in range(20):
        if status.state == "completed":
            break
        time.sleep(0.1)
        status = provider.get_status(handle)
    
    assert status.state == "completed"
    
    # 6. Retrieve output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_result = provider.retrieve_output(handle, tmpdir)
        assert output_result.job_id == "hf_job_123"
        assert ".safetensors" in output_result.output_path


def test_provider_validate_job():
    """Test job specification validation."""
    config = {}
    provider = FakeKaggleProvider(config)
    
    # Valid job
    valid_job = JobSpec(
        job_id="valid_job",
        name="Valid Job",
        provider="fake_kaggle",
        config={},
    )
    errors = provider.validate_job(valid_job)
    assert len(errors) == 0
    
    # Invalid job - empty job_id
    invalid_job = JobSpec(
        job_id="",
        name="Invalid Job",
        provider="fake_kaggle",
        config={},
    )
    errors = provider.validate_job(invalid_job)
    assert len(errors) > 0
    assert any("Job ID" in e for e in errors)
    
    # Invalid job - provider mismatch
    mismatch_job = JobSpec(
        job_id="mismatch",
        name="Mismatch Job",
        provider="wrong_provider",
        config={},
    )
    errors = provider.validate_job(mismatch_job)
    assert len(errors) > 0
    assert any("Provider mismatch" in e for e in errors)


def test_provider_manager_registration():
    """Test provider manager registration and retrieval."""
    config = Config()
    # Load YAML config files to get providers
    from pathlib import Path
    import yaml
    config_dir = Path("config")
    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            config.load_from_file(str(config_file), config_file.stem)
    
    manager = ProviderManager(config)
    
    # Get provider names
    names = manager.get_provider_names()
    assert "kaggle" in names
    assert "colab" in names
    assert "huggingface" in names
    
    # Get provider by name
    kaggle = manager.get_provider("kaggle")
    assert kaggle is not None
    assert kaggle.name == "kaggle"
    
    # Get non-existent provider
    nonexistent = manager.get_provider("nonexistent")
    assert nonexistent is None


def test_provider_manager_authenticate_all():
    """Test authenticating with all providers."""
    config = Config()
    from pathlib import Path
    config_dir = Path("config")
    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            config.load_from_file(str(config_file), config_file.stem)
    
    manager = ProviderManager(config)
    
    results = manager.authenticate()
    
    assert "kaggle" in results
    assert "colab" in results
    assert "huggingface" in results
    
    for name, result in results.items():
        assert result.provider == name
        # Real providers may not authenticate without credentials,
        # but the authenticate method should return a result
        assert isinstance(result.authenticated, bool)


def test_provider_manager_health_check():
    """Test provider health check."""
    config = Config()
    from pathlib import Path
    config_dir = Path("config")
    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            config.load_from_file(str(config_file), config_file.stem)
    
    manager = ProviderManager(config)
    
    health = manager.check_provider_health()
    
    assert "kaggle" in health
    assert "colab" in health
    assert "huggingface" in health
    
    for name, status in health.items():
        assert "authenticated" in status
        assert "available" in status


def test_provider_selection():
    """Test provider selection for a job using fake providers."""
    config = {}
    provider = FakeKaggleProvider(config)
    
    job_spec = JobSpec(
        job_id="selection_test",
        name="Selection Test",
        provider="fake_kaggle",
        config={"base_model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
    )
    
    # Authenticate first
    auth_result = provider.authenticate()
    assert auth_result.authenticated is True
    
    # Validate job
    errors = provider.validate_job(job_spec)
    assert len(errors) == 0
    
    # Check availability
    availability = provider.check_availability()
    assert availability.available is True
    
    # Submit job directly to provider
    handle = provider.submit_job(job_spec)
    assert handle.job_id == "selection_test"
    assert handle.provider == "fake_kaggle"


def test_provider_submit_job_through_manager():
    """Test job submission through provider manager with fake provider."""
    from nexus.providers.manager import ProviderManager
    
    config = Config()
    manager = ProviderManager(config)
    
    # Register fake provider directly
    fake_kaggle = FakeKaggleProvider({"timeout_seconds": 30, "max_retries": 3, "retry_delay": 1.0, "auth_env_vars": {}})
    manager._providers["fake_kaggle"] = fake_kaggle
    
    job_spec = JobSpec(
        job_id="manager_submit_test",
        name="Manager Submit Test",
        provider="fake_kaggle",
        config={"base_model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
    )
    
    handle = manager.submit_job(job_spec)
    assert handle.job_id == "manager_submit_test"
    assert handle.provider == "fake_kaggle"


def test_provider_wait_for_job():
    """Test waiting for job completion with fake provider."""
    config = Config()
    manager = ProviderManager(config)
    
    # Register fake provider directly
    fake_kaggle = FakeKaggleProvider({"timeout_seconds": 30, "max_retries": 3, "retry_delay": 1.0, "auth_env_vars": {}})
    manager._providers["fake_kaggle"] = fake_kaggle
    
    job_spec = JobSpec(
        job_id="wait_test",
        name="Wait Test",
        provider="fake_kaggle",
        config={},
    )
    
    handle = manager.submit_job(job_spec)
    status = manager.wait_for_job(handle, timeout=10.0, poll_interval=0.5)
    
    assert status.state == "completed"


def test_provider_cancel_job():
    """Test job cancellation through manager with fake provider."""
    config = Config()
    manager = ProviderManager(config)
    
    # Register fake provider directly
    fake_kaggle = FakeKaggleProvider({"timeout_seconds": 30, "max_retries": 3, "retry_delay": 1.0, "auth_env_vars": {}})
    manager._providers["fake_kaggle"] = fake_kaggle
    
    job_spec = JobSpec(
        job_id="cancel_test",
        name="Cancel Test",
        provider="fake_kaggle",
        config={},
    )
    
    handle = manager.submit_job(job_spec)
    
    # Cancel the job
    try:
        manager.cancel_job(handle)
        status = manager.get_job_status(handle)
        assert status.state == "cancelled"
    except ProviderError:
        # May not support cancellation - acceptable
        pass


def test_fake_provider_factory():
    """Test FakeProvider factory methods."""
    config = {}
    
    kaggle = FakeProvider.get_kaggle_provider(config)
    assert isinstance(kaggle, FakeKaggleProvider)
    
    colab = FakeProvider.get_colab_provider(config)
    assert isinstance(colab, FakeColabProvider)
    
    hf = FakeProvider.get_huggingface_provider(config)
    assert isinstance(hf, FakeHuggingFaceProvider)


def test_capabilities_dataclass():
    """Test Capabilities dataclass."""
    caps = Capabilities(
        gpu_memory_gb=14.56,
        max_runtime_h=168,
        resume=True,
        gpu_count=1,
        supports_colab=True,
        supports_huggingface=True,
    )
    
    assert caps.gpu_memory_gb == 14.56
    assert caps.max_runtime_h == 168
    assert caps.resume is True


def test_quota_info_dataclass():
    """Test QuotaInfo dataclass."""
    quota = QuotaInfo(
        gpu_type="Tesla T4",
        quota_type="gpu_hours",
        used=0.5,
        limit=30.0,
        remaining=29.5,
    )
    
    assert quota.used == 0.5
    assert quota.limit == 30.0
    assert quota.remaining == 29.5
    assert quota.used < quota.limit


def test_auth_result_dataclass():
    """Test AuthResult dataclass."""
    result = AuthResult(
        provider="test",
        method=AuthMethod.TOKEN,
        authenticated=True,
        token="secret_token",
    )
    
    assert result.authenticated is True
    assert result.method == AuthMethod.TOKEN
    assert result.token == "secret_token"


def test_job_spec_dataclass():
    """Test JobSpec dataclass."""
    job = JobSpec(
        job_id="test_123",
        name="Test Job",
        provider="kaggle",
        config={"key": "value"},
        priority=5,
    )
    
    assert job.job_id == "test_123"
    assert job.name == "Test Job"
    assert job.provider == "kaggle"
    assert job.config == {"key": "value"}
    assert job.priority == 5


def test_job_handle_dataclass():
    """Test JobHandle dataclass."""
    handle = JobHandle(
        job_id="test_123",
        provider="kaggle",
        handle="internal_handle_456",
    )
    
    assert handle.job_id == "test_123"
    assert handle.provider == "kaggle"
    assert handle.handle == "internal_handle_456"


def test_output_result_dataclass():
    """Test OutputResult dataclass."""
    result = OutputResult(
        job_id="test_123",
        provider="kaggle",
        output_path="/tmp/output",
        output_files=["file1.json", "file2.bin"],
    )
    
    assert result.job_id == "test_123"
    assert result.provider == "kaggle"
    assert len(result.output_files) == 2