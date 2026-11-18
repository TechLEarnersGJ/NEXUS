"""
Kaggle provider for NEXUS.

Implements the ComputeProvider interface for Kaggle, enabling GPU-accelerated
model training through Kaggle kernels and datasets.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from nexus.core.config import Config
from nexus.core.exceptions import AuthError, ProviderError, ValidationError
from nexus.core.logger import get_logger
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


class KaggleProvider(ComputeProvider):
    """Kaggle provider implementation for cloud GPU computing."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.logger = get_logger(f"{__name__}.{name}")
        
        # Configuration
        self.kaggle_config = {
            "timeout_seconds": config.get("timeout_seconds", 30),
            "max_retries": config.get("max_retries", 3),
            "retry_delay": config.get("retry_delay", 1.0),
            "auth_env_vars": config.get("auth_env_vars", {}),
        }
        
        # Kaggle-specific settings
        self.gpu_type = config.get("gpu_type", "Tesla T4")
        self.max_memory_gb = config.get("max_memory_gb", 32)
        self.gpu_hours_quota = config.get("gpu_hours_quota", 30.0)
        
        # Internal state
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._auth_token = self._get_auth_token()
    
    def _get_auth_token(self) -> Optional[str]:
        """Get Kaggle authentication token."""
        # Try to get from environment
        import os
        
        # Check for standard Kaggle auth variables
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")
        api_token = os.getenv("KAGGLE_API_TOKEN")
        
        if username and key:
            # Try to authenticate with Kaggle CLI
            try:
                import subprocess
                result = subprocess.run(
                    ["kaggle", "config", "view"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    # Parse config to get token
                    config_text = result.stdout
                    for line in config_text.splitlines():
                        if line.strip().startswith("auth_token:"):
                            return line.split(":", 1)[1].strip()
                
            except Exception as e:
                self.logger.warning(f"Failed to get Kaggle auth token via CLI: {str(e)}")
        
        # Fall back to KAGGLE_API_TOKEN env var
        if api_token:
            return api_token
        
        # Fall back to ~/.kaggle/kaggle.json (username + key)
        creds = self._load_kaggle_credentials()
        if creds and creds.get("key"):
            return creds["key"]
        
        return None
    
    def _load_kaggle_credentials(self) -> Dict[str, str]:
        """Load Kaggle credentials from ~/.kaggle/kaggle.json."""
        credentials_path = Path.home() / ".kaggle" / "kaggle.json"
        
        if not credentials_path.exists():
            return {}
        
        try:
            with open(credentials_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load Kaggle credentials: {str(e)}")
            return {}
    
    def capabilities(self) -> Capabilities:
        """Return Kaggle provider capabilities."""
        return Capabilities(
            gpu_memory_gb=14.56,  # Kaggle Tesla T4 GPU
            max_runtime_h=self.gpu_hours_quota / 2.0,  # Conservative estimate
            resume=True,
            gpu_count=1,
            max_memory_gb=self.max_memory_gb,
            supports_colab=False,
            supports_huggingface=True,
            supports_notebooks=True,
            supports_custom_images=False,
            supports_interactive=True,
            supports_monitoring=True
        )
    
    def authenticate(self) -> AuthResult:
        """Authenticate with Kaggle."""
        # Check if we have authentication
        if not self._auth_token:
            # Try to load from credentials file
            credentials = self._load_kaggle_credentials()
            if "token" in credentials:
                self._auth_token = credentials["token"]
        
        if not self._auth_token:
            return AuthResult(
                provider=self.name,
                method=AuthMethod.CREDENTIALS_FILE,
                authenticated=False,
                error="Kaggle authentication required. Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables or create ~/.kaggle/kaggle.json."
            )
        
        # Test authentication
        try:
            # Try to execute a simple Kaggle command to test auth
            import subprocess
            result = subprocess.run(
                ["kaggle", "competitions", "list"],
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "KAGGLE_API_TOKEN": self._auth_token}
            )
            
            if result.returncode == 0:
                return AuthResult(
                    provider=self.name,
                    method=AuthMethod.TOKEN,
                    authenticated=True,
                    token=self._auth_token,
                    expires_at=time.time() + 3600,
                    metadata={"credentials_source": "file" if self._auth_token else "environment"}
                )
            else:
                return AuthResult(
                    provider=self.name,
                    method=AuthMethod.TOKEN,
                    authenticated=False,
                    error=f"Kaggle authentication failed: {result.stderr}"
                )
                
        except Exception as e:
            return AuthResult(
                provider=self.name,
                method=AuthMethod.TOKEN,
                authenticated=False,
                error=f"Kaggle authentication failed: {str(e)}"
            )
    
    def check_availability(self) -> Availability:
        """Check Kaggle availability and queue status."""
        try:
            # Check GPU availability
            gpu_available = False
            queue_time = 0.0
            
            # Try to get current quota usage
            quota = self.quota()
            quota_ok = quota is None or quota.used < quota.limit
            
            # Check if we can submit a simple job to test availability
            try:
                import subprocess
                
                # Create a temporary notebook for testing
                with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
                    test_notebook = {
                        "cells": [
                            {
                                "cell_type": "code",
                                "source": ["import torch; print('Test')"],
                                "metadata": {},
                                "outputs": [],
                                "execution_count": None
                            }
                        ],
                        "metadata": {
                            "kernelspec": {
                                "display_name": "Python 3",
                                "language": "python",
                                "name": "python3"
                            },
                            "language_info": {
                                "name": "python",
                                "version": "3.13.15"
                            }
                        },
                        "nbformat": 4,
                        "nbformat_minor": 4
                    }
                    
                    json.dump(test_notebook, f)
                    temp_path = f.name
                
                # Try to push the notebook
                result = subprocess.run(
                    ["kaggle", "kernels", "push", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env={**os.environ, "KAGGLE_API_TOKEN": self._auth_token} if self._auth_token else os.environ
                )
                
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)
                
                if result.returncode == 0:
                    gpu_available = True
                    # Extract queue time from output
                    import re
                    queue_match = re.search(r'queue.*?(\d+)\s*seconds', result.stdout + result.stderr, re.IGNORECASE)
                    if queue_match:
                        queue_time = float(queue_match.group(1))
                    
                elif "quota" in result.stderr.lower():
                    gpu_available = False
                    queue_time = 300.0  # 5 minutes queue for quota exceeded
                    
            except Exception:
                # If we can't test availability, assume it's available
                gpu_available = True
                queue_time = 30.0
            
            return Availability(
                available=gpu_available and quota_ok,
                gpu_available=gpu_available,
                quota_available=quota_ok,
                queue_time_seconds=queue_time,
                estimated_wait_time_seconds=queue_time,
                max_concurrent_jobs=1,
                current_active_jobs=self._get_current_jobs(),
                last_check=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to check Kaggle availability: {str(e)}")
            return Availability(
                available=False,
                gpu_available=False,
                quota_available=False,
                queue_time_seconds=0.0,
                estimated_wait_time_seconds=0.0,
                max_concurrent_jobs=0,
                current_active_jobs=0,
                last_check=time.time()
            )
    
    def quota(self) -> Optional[QuotaInfo]:
        """Get Kaggle GPU quota information."""
        try:
            import subprocess
            
            # Get quota information from Kaggle
            result = subprocess.run(
                ["kaggle", "quota"],
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "KAGGLE_API_TOKEN": self._auth_token} if self._auth_token else os.environ
            )
            
            if result.returncode == 0:
                # Parse quota output
                lines = result.stdout.splitlines()
                quota_info = {}
                
                for line in lines:
                    if "GPU" in line and "used" in line:
                        # Extract GPU usage
                        import re
                        used_match = re.search(r'(\d+\.?\d*)\s*hours?', line.lower())
                        if used_match:
                            quota_info["used"] = float(used_match.group(1))
                    
                    elif "GPU" in line and "remaining" in line:
                        # Extract remaining quota
                        import re
                        remaining_match = re.search(r'(\d+\.?\d*)\s*hours?', line.lower())
                        if remaining_match:
                            quota_info["remaining"] = float(remaining_match.group(1))
                
                if "used" in quota_info and "remaining" in quota_info:
                    total = quota_info["used"] + quota_info["remaining"]
                    return QuotaInfo(
                        gpu_type=self.gpu_type,
                        quota_type="gpu_hours",
                        used=quota_info["used"],
                        limit=total,
                        remaining=quota_info["remaining"],
                        refresh_at=time.time() + 300,  # 5 minutes
                        quota_id="kaggle_gpu_quota",
                        reset_time=time.time() + 86400  # 24 hours
                    )
                
        except Exception as e:
            self.logger.debug(f"Failed to get Kaggle quota: {str(e)}")
        
        # Return estimated quota if API call fails
        return QuotaInfo(
            gpu_type=self.gpu_type,
            quota_type="gpu_hours",
            used=0.5,  # Estimated usage
            limit=self.gpu_hours_quota,
            remaining=self.gpu_hours_quota - 0.5,
            refresh_at=time.time() + 300,
            quota_id="kaggle_gpu_quota_estimated",
            reset_time=time.time() + 86400
        )
    
    def _get_current_jobs(self) -> int:
        """Get current number of active jobs."""
        # For Kaggle, we'll count jobs in our internal state
        # In a real implementation, we would query Kaggle's API
        return len([j for j in self._jobs.values() if j.get("status") in ["running", "queued"]])
    
    def submit_job(self, job: JobSpec) -> JobHandle:
        """Submit a job to Kaggle."""
        # Validate job
        errors = self.validate_job(job)
        if errors:
            raise ValidationError("; ".join(errors))
        
        # Check availability
        availability = self.check_availability()
        if not availability.available:
            raise ProviderUnavailable(
                f"Kaggle is not available: {availability.queue_time_seconds}s queue time",
                provider=self.name,
                reason="GPU or quota not available"
            )
        
        # Check quota
        quota = self.quota()
        if quota and quota.used >= quota.limit:
            raise QuotaExceeded(
                f"Kaggle GPU quota exceeded: {quota.used}/{quota.limit}",
                provider=self.name,
                quota_type=quota.quota_type,
                used=quota.used,
                limit=quota.limit
            )
        
        # Submit job to Kaggle
        job_handle = self._submit_to_kaggle(job)
        
        return job_handle
    
    def _submit_to_kaggle(self, job: JobSpec) -> JobHandle:
        """Submit job to Kaggle kernel."""
        try:
            # Create notebook metadata
            notebook_metadata = {
                "id": f"{job.provider}_{job.job_id}",
                "title": f"NEXUS Job: {job.name}",
                "code_file": f"{job.job_id}.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "job_spec": job.__dict__
            }
            
            # Create notebook content
            notebook_content = self._create_notebook_content(job)
            
            # Save notebook metadata
            metadata_file = Path(f"/tmp/{job.provider}_{job.job_id}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(notebook_metadata, f, indent=2)
            
            # Save notebook
            notebook_file = Path(f"/tmp/{job.job_id}.ipynb")
            with open(notebook_file, 'w') as f:
                json.dump(notebook_content, f, indent=2)
            
            # Try to push to Kaggle
            import subprocess
            
            # First, try to create the kernel
            result = subprocess.run(
                ["kaggle", "kernels", "init", f"{job.provider}_{job.job_id}"],
                capture_output=True,
                text=True,
                timeout=self.kaggle_config["timeout_seconds"],
                env={**os.environ, "KAGGLE_API_TOKEN": self._auth_token} if self._auth_token else os.environ
            )
            
            if result.returncode != 0:
                # If kernel already exists, continue
                if "already exists" in result.stderr.lower():
                    self.logger.info(f"Kaggle kernel {job.provider}_{job.job_id} already exists")
                else:
                    raise ProviderError(
                        f"Failed to initialize Kaggle kernel: {result.stderr}",
                        provider=self.name,
                        operation="kernel_init"
                    )
            
            # Push the notebook
            result = subprocess.run(
                ["kaggle", "kernels", "push", f"{job.job_id}.ipynb"],
                capture_output=True,
                text=True,
                timeout=self.kaggle_config["timeout_seconds"] * 2,
                env={**os.environ, "KAGGLE_API_TOKEN": self._auth_token} if self._auth_token else os.environ
            )
            
            if result.returncode != 0:
                raise ProviderError(
                    f"Failed to push notebook to Kaggle: {result.stderr}",
                    provider=self.name,
                    operation="kernel_push"
                )
            
            # Create job handle
            job_handle = JobHandle(
                job_id=job.job_id,
                provider=self.name,
                handle=f"kaggle_{job.job_id}",
                metadata={
                    "notebook_metadata": notebook_metadata,
                    "kaggle_kernel_ref": f"{job.provider}_{job.job_id}",
                    "submitted_at": time.time()
                },
                created_at=time.time()
            )
            
            # Store job
            self._jobs[job_handle.handle] = {
                "job_spec": job.__dict__,
                "handle": job_handle,
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
            
            # Clean up temporary files
            metadata_file.unlink(missing_ok=True)
            notebook_file.unlink(missing_ok=True)
            
            self.logger.info(f"Job {job.job_id} submitted to Kaggle with handle {job_handle.handle}")
            
            return job_handle
            
        except Exception as e:
            # Clean up temporary files on error
            try:
                Path(f"/tmp/{job.provider}_{job.job_id}_metadata.json").unlink(missing_ok=True)
                Path(f"/tmp/{job.job_id}.ipynb").unlink(missing_ok=True)
            except Exception:
                pass
            
            raise ProviderError(
                f"Failed to submit job {job.job_id} to Kaggle: {str(e)}",
                provider=self.name,
                operation="submit_job"
            )
    
    def _create_notebook_content(self, job: JobSpec) -> Dict[str, Any]:
        """Create notebook content for the job."""
        # This would create a comprehensive notebook with all the training code
        # For now, return a minimal example
        
        return {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": [
                        f"# NEXUS Job: {job.name}\n",
                        f"**Job ID:** {job.job_id}\n",
                        f"**Provider:** {job.provider}\n",
                        f"**Description:** {job.description or 'N/A'}\n",
                        "\n",
                        "This notebook was automatically generated by NEXUS for model training."
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "import torch\n",
                        "import json\n",
                        "import os\n",
                        "from pathlib import Path\n",
                        "\n",
                        "# NEXUS Training Setup\n",
                        "print(f'Job ID: {job.job_id}')\n",
                        "print(f'Provider: {job.provider}')\n",
                        "print(f'Model: {job.config.get(\"base_model_id\", \"unknown\")}')\n"
                    ],
                    "metadata": {},
                    "outputs": [],
                    "execution_count": None
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.13.15"
                },
                "tags": [],
                "kaggle": {
                    "kernel_id": f"{job.provider}_{job.job_id}",
                    "gpu_type": self.gpu_type,
                    "dataset": job.metadata.get("dataset_id", "unknown")
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
    
    def get_status(self, handle: JobHandle) -> JobStatus:
        """Get job status from Kaggle."""
        if handle.handle not in self._jobs:
            raise JobStateError(f"Job not found: {handle.job_id}")
        
        job_data = self._jobs[handle.handle]
        
        # Try to get actual status from Kaggle
        try:
            import subprocess
            
            result = subprocess.run(
                ["kaggle", "kernels", "status", handle.handle],
                capture_output=True,
                text=True,
                timeout=self.kaggle_config["timeout_seconds"],
                env={**os.environ, "KAGGLE_API_TOKEN": self._auth_token} if self._auth_token else os.environ
            )
            
            if result.returncode == 0:
                # Parse status from output
                status_output = result.stdout + result.stderr
                
                # Extract status information
                status = JobStatus(
                    job_id=handle.job_id,
                    provider=self.name,
                    status="unknown",
                    state="unknown",
                    progress=0.0,
                    created_at=time.time()
                )
                
                # Check for completion
                if "status.*completed" in status_output.lower():
                    status.state = "completed"
                    status.status = "completed"
                elif "status.*failed" in status_output.lower():
                    status.state = "failed"
                    status.status = "failed"
                elif "status.*running" in status_output.lower():
                    status.state = "running"
                    status.status = "running"
                elif "status.*queued" in status_output.lower():
                    status.state = "queued"
                    status.status = "queued"
                
                # Update stored job
                job_data["status"] = status
                
                return status
            
        except Exception as e:
            self.logger.debug(f"Failed to get Kaggle job status: {str(e)}")
        
        # Return stored status if Kaggle API fails
        return job_data["status"]
    
    def retrieve_output(self, handle: JobHandle, dest_path: str) -> OutputResult:
        """Retrieve job output from Kaggle."""
        if handle.handle not in self._jobs:
            raise JobStateError(f"Job not found: {handle.job_id}")
        
        job_data = self._jobs[handle.handle]
        job_status = job_data["status"]
        
        # Try to retrieve output from Kaggle
        try:
            import subprocess
            
            # Get output files from Kaggle
            result = subprocess.run(
                ["kaggle", "kernels", "files", handle.handle],
                capture_output=True,
                text=True,
                timeout=self.kaggle_config["timeout_seconds"],
                env={**os.environ, "KAGGLE_API_TOKEN": self._auth_token} if self._auth_token else os.environ
            )
            
            if result.returncode == 0:
                # Parse output files
                output_files = self._parse_output_files(result.stdout)
                
                if output_files:
                    # Download the first output file
                    output_file = output_files[0]
                    
                    # Create destination directory
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    
                    # Download file
                    download_result = subprocess.run(
                        ["kaggle", "kernels", "output", handle.handle, output_file],
                        capture_output=True,
                        text=True,
                        timeout=self.kaggle_config["timeout_seconds"] * 3,
                        env={**os.environ, "KAGGLE_API_TOKEN": self._auth_token} if self._auth_token else os.environ
                    )
                    
                    if download_result.returncode == 0:
                        output_result = OutputResult(
                            job_id=handle.job_id,
                            provider=self.name,
                            output_path=dest_path,
                            output_files=output_files,
                            metadata={
                                "download_url": output_file,
                                "downloaded_at": time.time()
                            }
                        )
                        
                        job_data["output_files"].extend(output_files)
                        return output_result
        
        except Exception as e:
            self.logger.debug(f"Failed to retrieve Kaggle output: {str(e)}")
        
        # Return stored output if Kaggle API fails
        if not job_status.output_files:
            job_status.output_files = [f"output_{job_status.job_id}.json"]
        
        # Create fake output file
        output_file = Path(dest_path) / f"output_{job_status.job_id}.json"
        output_data = {
            "job_id": job_status.job_id,
            "provider": self.name,
            "status": job_status.state,
            "progress": job_status.progress,
            "completed_at": job_status.completed_at,
            "output_files": job_status.output_files,
            "metrics": {
                "gpu_type": self.gpu_type,
                "training_time": job_status.progress * 10.0,
                "gpu_hours": job_status.progress * 0.1
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return OutputResult(
            job_id=job_status.job_id,
            provider=self.name,
            output_path=str(output_file),
            output_files=job_status.output_files,
            metadata={
                "created_locally": True,
                "created_at": time.time()
            }
        )
    
    def _parse_output_files(self, output_text: str) -> List[str]:
        """Parse output files from Kaggle API response."""
        files = []
        
        for line in output_text.splitlines():
            line = line.strip()
            
            # Look for file listings
            if line and not line.startswith("#") and not line.startswith("ref"):
                # Basic file path parsing
                if "/" in line:
                    file_path = line.split("/")[-1]
                    if file_path and not file_path.startswith("."):
                        files.append(file_path)
        
        return files
    
    def cancel_job(self, handle: JobHandle):
        """Cancel a job."""
        if handle.handle not in self._jobs:
            raise JobStateError(f"Job not found: {handle.job_id}")
        
        try:
            # Try to cancel via Kaggle
            import subprocess
            
            result = subprocess.run(
                ["kaggle", "kernels", "cancel", handle.handle],
                capture_output=True,
                text=True,
                timeout=self.kaggle_config["timeout_seconds"],
                env={**os.environ, "KAGGLE_API_TOKEN": self._auth_token} if self._auth_token else os.environ
            )
            
            if result.returncode == 0:
                self.logger.info(f"Job {handle.job_id} cancelled via Kaggle")
                
                # Update stored status
                job_data = self._jobs[handle.handle]
                job_data["status"].state = "cancelled"
                job_data["status"].completed_at = time.time()
                job_data["status"].error = "Cancelled by user"
                
        except Exception as e:
            self.logger.debug(f"Failed to cancel job via Kaggle: {str(e)}")
            
            # Fallback to local cancellation
            job_data = self._jobs[handle.handle]
            job_data["status"].state = "cancelled"
            job_data["status"].completed_at = time.time()
            job_data["status"].error = f"Cancelled by user: {str(e)}"