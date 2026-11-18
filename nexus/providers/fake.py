"""
Fake providers for testing.

Provides mock implementations of cloud providers for testing and development
without requiring actual cloud services.
"""

import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

from nexus.core.exceptions import (
    AuthError,
    ProviderError,
    ProviderUnavailable,
    QuotaExceeded,
    JobStateError,
)
from nexus.providers.base import (
    ComputeProvider,
    JobSpec,
    JobHandle,
    JobStatus,
    AuthResult,
    OutputResult,
    Capabilities,
    Availability,
    QuotaInfo,
    AuthMethod,
)


class FakeKaggleProvider(ComputeProvider):
    """Fake Kaggle provider for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("fake_kaggle", config)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._outputs: Dict[str, Dict[str, Any]] = {}
        self._auth_token = config.get("auth_token", "fake_token")
    
    def capabilities(self) -> Capabilities:
        return Capabilities(
            gpu_memory_gb=14.56,
            max_runtime_h=168,
            resume=True,
            gpu_count=1,
            max_memory_gb=32,
            supports_colab=False,
            supports_huggingface=True,
            supports_notebooks=True,
            supports_custom_images=False,
            supports_interactive=True,
            supports_monitoring=True
        )
    
    def authenticate(self) -> AuthResult:
        # Simulate authentication check
        if self._auth_token == "fake_token":
            return AuthResult(
                provider=self.name,
                method=AuthMethod.TOKEN,
                authenticated=True,
                token=self._auth_token,
                expires_at=time.time() + 3600
            )
        else:
            return AuthResult(
                provider=self.name,
                method=AuthMethod.TOKEN,
                authenticated=False,
                error="Invalid authentication token"
            )
    
    def check_availability(self) -> Availability:
        # Simulate availability checks
        return Availability(
            available=True,
            gpu_available=True,
            quota_available=True,
            queue_time_seconds=0.0,
            estimated_wait_time_seconds=0.0,
            max_concurrent_jobs=10,
            current_active_jobs=len([j for j in self._jobs.values() if j.get("status") in ["running", "queued"]]),
            last_check=time.time()
        )
    
    def quota(self) -> Optional[QuotaInfo]:
        # Simulate quota information
        return QuotaInfo(
            gpu_type="Tesla T4",
            quota_type="gpu_hours",
            used=0.5,
            limit=30.0,
            remaining=29.5,
            refresh_at=time.time() + 300,
            quota_id="kaggle_gpu_quota",
            reset_time=time.time() + 86400
        )
    
    def submit_job(self, job: JobSpec) -> JobHandle:
        # Validate job
        errors = self.validate_job(job)
        if errors:
            raise ValidationError("; ".join(errors))
        
        # Generate job handle
        handle = JobHandle(
            job_id=job.job_id,
            provider=self.name,
            handle=f"fake_handle_{uuid.uuid4().hex[:8]}",
            metadata={
                "job_spec": job.__dict__,
                "submitted_at": time.time()
            },
            created_at=time.time()
        )
        
        # Store job
        self._jobs[handle.handle] = {
            "job_spec": job.__dict__,
            "handle": handle,
            "status": JobStatus(
                job_id=job.job_id,
                provider=self.name,
                status="pending",
                state="pending",
                progress=0.0,
                created_at=time.time()
            ),
            "output_files": [],
            "started_at": None,
            "completed_at": None,
            "error": None
        }
        
        return handle
    
    def get_status(self, handle: JobHandle) -> JobStatus:
        if handle.handle not in self._jobs:
            raise JobStateError(f"Job not found: {handle.job_id}")
        
        job_data = self._jobs[handle.handle]
        
        # Simulate job progress
        job_status = job_data["status"]
        
        if job_status.state == "pending":
            # Simulate job starting
            job_status.state = "running"
            job_status.started_at = time.time()
            job_status.progress = 0.0
        
        if job_status.state == "running":
            # Simulate job progress
            job_status.progress = min(100.0, job_status.progress + 10.0)
            
            # Simulate completion after some time
            if job_status.progress >= 100.0 and job_status.started_at:
                job_status.state = "completed"
                job_status.completed_at = time.time()
                job_status.progress = 100.0
                job_status.output_files = [f"output_{job_status.job_id}.json"]
        
        return job_status
    
    def retrieve_output(self, handle: JobHandle, dest_path: str) -> OutputResult:
        if handle.handle not in self._jobs:
            raise JobStateError(f"Job not found: {handle.job_id}")
        
        job_data = self._jobs[handle.handle]
        job_status = job_data["status"]
        
        if job_status.state != "completed":
            raise JobStateError(f"Job {handle.job_id} is not completed")
        
        # Create output directory
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        
        # Create fake output file
        output_file = Path(dest_path) / f"output_{handle.job_id}.json"
        output_data = {
            "job_id": handle.job_id,
            "status": job_status.state,
            "progress": job_status.progress,
            "completed_at": job_status.completed_at,
            "output_files": job_status.output_files,
            "metrics": {
                "loss": 0.123,
                "accuracy": 0.876,
                "perplexity": 2.5
            }
        }
        
        with open(output_file, 'w') as f:
            import json
            json.dump(output_data, f, indent=2)
        
        # Store output info
        self._outputs[handle.handle] = {
            "output_path": str(output_file),
            "output_files": job_status.output_files,
            "downloaded_at": time.time(),
            "model": {
                "size": "0.5B",
                "architecture": "Qwen2.5",
                "quantization": "4-bit",
            }
        }
        
        return OutputResult(
            job_id=handle.job_id,
            provider=self.name,
            output_path=str(output_file),
            output_files=job_status.output_files,
            metadata={
                "job_status": job_status.__dict__,
                "downloaded_at": time.time(),
                "model": {
                    "size": "0.5B",
                    "architecture": "Qwen2.5",
                    "quantization": "4-bit",
                }
            }
        )

    def cancel_job(self, handle: JobHandle):
        if handle.handle not in self._jobs:
            raise JobStateError(f"Job not found: {handle.job_id}")
        
        job_data = self._jobs[handle.handle]
        job_status = job_data["status"]
        
        if job_status.state in ["completed", "failed"]:
            raise JobStateError(f"Cannot cancel completed job: {handle.job_id}")
        
        job_status.state = "cancelled"
        job_status.completed_at = time.time()
        job_status.error = "Job cancelled"


class FakeColabProvider(ComputeProvider):
    """Fake Colab provider for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("fake_colab", config)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._outputs: Dict[str, Dict[str, Any]] = {}
        self._auth_token = config.get("auth_token", "fake_colab_token")
    
    def capabilities(self) -> Capabilities:
        return Capabilities(
            gpu_memory_gb=12,
            max_runtime_h=12,
            resume=False,
            gpu_count=1,
            max_memory_gb=24,
            supports_colab=True,
            supports_huggingface=True,
            supports_notebooks=True,
            supports_custom_images=False,
            supports_interactive=True,
            supports_monitoring=True
        )
    
    def authenticate(self) -> AuthResult:
        if self._auth_token == "fake_colab_token":
            return AuthResult(
                provider=self.name,
                method=AuthMethod.OAUTH,
                authenticated=True,
                token=self._auth_token,
                expires_at=time.time() + 3600
            )
        else:
            return AuthResult(
                provider=self.name,
                method=AuthMethod.OAUTH,
                authenticated=False,
                error="Invalid authentication token"
            )
    
    def check_availability(self) -> Availability:
        return Availability(
            available=True,
            gpu_available=True,
            quota_available=True,
            queue_time_seconds=0.0,
            estimated_wait_time_seconds=0.0,
            max_concurrent_jobs=1,
            current_active_jobs=len([j for j in self._jobs.values() if j.get("status") in ["running", "queued"]]),
            last_check=time.time()
        )
    
    def quota(self) -> Optional[QuotaInfo]:
        return QuotaInfo(
            gpu_type="Tesla T4",
            quota_type="runtime",
            used=0.1,
            limit=12.0,
            remaining=11.9,
            refresh_at=time.time() + 300,
            quota_id="colab_runtime_quota",
            reset_time=time.time() + 86400
        )
    
    def submit_job(self, job: JobSpec) -> JobHandle:
        errors = self.validate_job(job)
        if errors:
            raise ValidationError("; ".join(errors))
        
        handle = JobHandle(
            job_id=job.job_id,
            provider=self.name,
            handle=f"fake_colab_handle_{uuid.uuid4().hex[:8]}",
            metadata={
                "job_spec": job.__dict__,
                "submitted_at": time.time(),
                "requires_gpu": True
            },
            created_at=time.time()
        )
        
        self._jobs[handle.handle] = {
            "job_spec": job.__dict__,
            "handle": handle,
            "status": JobStatus(
                job_id=job.job_id,
                provider=self.name,
                status="queued",
                state="queued",
                progress=0.0,
                created_at=time.time()
            ),
            "output_files": [],
            "started_at": None,
            "completed_at": None,
            "error": None
        }
        
        return handle
    
    def get_status(self, handle: JobHandle) -> JobStatus:
        if handle.handle not in self._jobs:
            raise JobStateError(f"Job not found: {handle.job_id}")
        
        job_data = self._jobs[handle.handle]
        job_status = job_data["status"]
        
        if job_status.state == "queued":
            # Simulate job starting
            job_status.state = "running"
            job_status.started_at = time.time()
            job_status.progress = 0.0
        
        if job_status.state == "running":
            job_status.progress = min(100.0, job_status.progress + 15.0)
            
            if job_status.progress >= 100.0 and job_status.started_at:
                job_status.state = "completed"
                job_status.completed_at = time.time()
                job_status.progress = 100.0
                job_status.output_files = [f"colab_output_{job_status.job_id}.json"]
        
        return job_status
    
    def retrieve_output(self, handle: JobHandle, dest_path: str) -> OutputResult:
        if handle.handle not in self._jobs:
            raise JobStateError(f"Job not found: {handle.job_id}")
        
        job_data = self._jobs[handle.handle]
        job_status = job_data["status"]
        
        if job_status.state != "completed":
            raise JobStateError(f"Job {handle.job_id} is not completed")
        
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        
        output_file = Path(dest_path) / f"colab_output_{handle.job_id}.json"
        output_data = {
            "job_id": handle.job_id,
            "provider": "colab",
            "status": job_status.state,
            "progress": job_status.progress,
            "completed_at": job_status.completed_at,
            "metrics": {
                "training_time": 120.5,
                "gpu_hours": 2.5,
                "loss": 0.098,
                "accuracy": 0.921,
                "perplexity": 1.8
            },
            "model": {
                "size": "0.5B",
                "architecture": "Qwen2.5",
                "quantization": "4-bit"
            }
        }
        
        with open(output_file, 'w') as f:
            import json
            json.dump(output_data, f, indent=2)
        
        self._outputs[handle.handle] = {
            "output_path": str(output_file),
            "output_files": job_status.output_files,
            "downloaded_at": time.time(),
            "model": {
                "size": "0.5B",
                "architecture": "Qwen2.5",
                "quantization": "4-bit",
            }
        }
        
        return OutputResult(
            job_id=handle.job_id,
            provider=self.name,
            output_path=str(output_file),
            output_files=job_status.output_files,
            metadata={
                "job_status": job_status.__dict__,
                "downloaded_at": time.time(),
                "model": {
                    "size": "0.5B",
                    "architecture": "Qwen2.5",
                    "quantization": "4-bit",
                }
            }
        )

    def cancel_job(self, handle: JobHandle):
        if handle.handle not in self._jobs:
            raise JobStateError(f"Job not found: {handle.job_id}")
        
        job_data = self._jobs[handle.handle]
        job_status = job_data["status"]
        
        if job_status.state in ["completed", "failed"]:
            raise JobStateError(f"Cannot cancel completed job: {handle.job_id}")
        
        job_status.state = "cancelled"
        job_status.completed_at = time.time()
        job_status.error = "Job cancelled"


class FakeHuggingFaceProvider(ComputeProvider):
    """Fake Hugging Face provider for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("fake_huggingface", config)
        self._models = {}
        self._datasets = {}
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._outputs: Dict[str, Dict[str, Any]] = {}
        self._auth_token = config.get("auth_token", "fake_hf_token")
    
    def capabilities(self) -> Capabilities:
        return Capabilities(
            gpu_memory_gb=0,
            max_runtime_h=0,
            resume=True,
            gpu_count=0,
            max_memory_gb=0,
            supports_colab=False,
            supports_huggingface=True,
            supports_notebooks=False,
            supports_custom_images=False,
            supports_interactive=False,
            supports_monitoring=False
        )
    
    def authenticate(self) -> AuthResult:
        if self._auth_token == "fake_hf_token":
            return AuthResult(
                provider=self.name,
                method=AuthMethod.TOKEN,
                authenticated=True,
                token=self._auth_token,
                expires_at=time.time() + 3600
            )
        else:
            return AuthResult(
                provider=self.name,
                method=AuthMethod.TOKEN,
                authenticated=False,
                error="Invalid authentication token"
            )
    
    def check_availability(self) -> Availability:
        return Availability(
            available=True,
            gpu_available=False,
            quota_available=True,
            queue_time_seconds=0.0,
            estimated_wait_time_seconds=0.0,
            max_concurrent_jobs=0,
            current_active_jobs=0,
            last_check=time.time()
        )
    
    def quota(self) -> Optional[QuotaInfo]:
        return QuotaInfo(
            gpu_type="",
            quota_type="api_calls",
            used=10.0,
            limit=60.0,
            remaining=50.0,
            refresh_at=time.time() + 60,
            quota_id="hf_api_quota",
            reset_time=time.time() + 60
        )
    
    def submit_job(self, job: JobSpec) -> JobHandle:
        errors = self.validate_job(job)
        if errors:
            raise ValidationError("; ".join(errors))
        
        handle = JobHandle(
            job_id=job.job_id,
            provider=self.name,
            handle=f"fake_hf_handle_{uuid.uuid4().hex[:8]}",
            metadata={
                "job_spec": job.__dict__,
                "submitted_at": time.time(),
                "model_registry": True
            },
            created_at=time.time()
        )
        
        self._jobs[handle.handle] = {
            "job_spec": job.__dict__,
            "handle": handle,
            "status": JobStatus(
                job_id=job.job_id,
                provider=self.name,
                status="pending",
                state="pending",
                progress=0.0,
                created_at=time.time()
            ),
            "output_files": [],
            "started_at": None,
            "completed_at": None,
            "error": None
        }
        
        return handle
    
    def get_status(self, handle: JobHandle) -> JobStatus:
        if handle.handle not in self._jobs:
            raise JobStateError(f"Job not found: {handle.job_id}")
        
        job_data = self._jobs[handle.handle]
        job_status = job_data["status"]
        
        if job_status.state == "pending":
            job_status.state = "completed"
            job_status.started_at = time.time()
            job_status.completed_at = time.time()
            job_status.progress = 100.0
            job_status.output_files = [f"model_{job_status.job_id}.safetensors"]
            job_status.metrics = {
                "model_size": "0.5B",
                "training_time": 45.0,
                "final_loss": 0.234,
                "final_accuracy": 0.867
            }
        
        return job_status
    
    def retrieve_output(self, handle: JobHandle, dest_path: str) -> OutputResult:
        if handle.handle not in self._jobs:
            raise JobStateError(f"Job not found: {handle.job_id}")
        
        job_data = self._jobs[handle.handle]
        job_status = job_data["status"]
        
        if job_status.state != "completed":
            raise JobStateError(f"Job {handle.job_id} is not completed")
        
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        
        output_file = Path(dest_path) / f"model_{handle.job_id}.safetensors"
        output_data = {
            "model_info": {
                "id": handle.job_id,
                "model_id": f"fake_user/{handle.job_id}",
                "architecture": "Qwen2.5",
                "parameters": 500000000,
                "quantization": "4-bit",
                "license": "apache-2.0",
                "size": "0.5B",
                "context_length": 2048,
                "vocab_size": 151936
            },
            "training_info": {
                "job_id": handle.job_id,
                "provider": self.name,
                "status": job_status.state,
                "progress": job_status.progress,
                "completed_at": job_status.completed_at,
                "metrics": job_status.metrics,
                "config": job_data["job_spec"].get("config", {})
            }
        }
        
        with open(output_file, 'w') as f:
            import json
            json.dump(output_data, f, indent=2)
        
        self._outputs[handle.handle] = {
            "output_path": str(output_file),
            "output_files": job_status.output_files,
            "downloaded_at": time.time(),
            "model": {
                "size": "0.5B",
                "architecture": "Qwen2.5",
                "quantization": "4-bit",
            }
        }
        
        return OutputResult(
            job_id=handle.job_id,
            provider=self.name,
            output_path=str(output_file),
            output_files=job_status.output_files,
            metadata={
                "job_status": job_status.__dict__,
                "downloaded_at": time.time(),
                "model": {
                    "size": "0.5B",
                    "architecture": "Qwen2.5",
                    "quantization": "4-bit",
                }
            }
        )

    def cancel_job(self, handle: JobHandle):
        if handle.handle not in self._jobs:
            raise JobStateError(f"Job not found: {handle.job_id}")
        
        job_data = self._jobs[handle.handle]
        job_status = job_data["status"]
        
        if job_status.state in ["completed", "failed"]:
            raise JobStateError(f"Cannot cancel completed job: {handle.job_id}")
        
        job_status.state = "cancelled"
        job_status.completed_at = time.time()
        job_status.error = "Job cancelled"


class FakeProvider:
    """Fake provider that returns fake implementations for testing."""
    
    @staticmethod
    def get_kaggle_provider(config: Dict[str, Any]) -> ComputeProvider:
        """Get a fake Kaggle provider."""
        return FakeKaggleProvider(config)
    
    @staticmethod
    def get_colab_provider(config: Dict[str, Any]) -> ComputeProvider:
        """Get a fake Colab provider."""
        return FakeColabProvider(config)
    
    @staticmethod
    def get_huggingface_provider(config: Dict[str, Any]) -> ComputeProvider:
        """Get a fake Hugging Face provider."""
        return FakeHuggingFaceProvider(config)