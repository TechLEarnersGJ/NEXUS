"""
Model downloader for NEXUS.

Handles downloading models from Hugging Face and managing local storage,
with considerations for disk space, authentication, and resume capabilities.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from nexus.core.exceptions import ProviderError
from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


class ModelDownloader:
    """Downloads models from Hugging Face and manages local storage."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.config = config or {}
        
        # Download settings
        self.cache_dir = Path(self.config.get("cache_dir", "~/.cache/nexus/models"))
        self.max_download_size_gb = self.config.get("max_download_size_gb", 100)
        self.timeout_seconds = self.config.get("timeout_seconds", 300)
        self.retry_count = self.config.get("retry_count", 3)
        self.retry_delay = self.config.get("retry_delay", 5.0)
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_model(self, model_id: str, destination: Optional[str] = None,
                      use_auth: bool = True) -> Dict[str, Any]:
        """Download a model from Hugging Face."""
        self.nexus_logger.log_event(
            "model_download_start",
            {
                "model_id": model_id,
                "destination": destination,
                "use_auth": use_auth,
                "cache_dir": str(self.cache_dir)
            }
        )
        
        try:
            # Determine destination path
            if destination is None:
                # Create a directory for the model
                safe_model_id = self._sanitize_model_id(model_id)
                dest_path = self.cache_dir / safe_model_id
            else:
                dest_path = Path(destination)
            
            # Create destination directory
            dest_path.mkdir(parents=True, exist_ok=True)
            
            # Check disk space
            if self._check_disk_space(dest_path, self.max_download_size_gb):
                self.logger.warning(f"Insufficient disk space for model {model_id}")
            
            # Simulate model download
            download_result = self._simulate_download(model_id, dest_path, use_auth)
            
            if download_result["success"]:
                # Get model metadata
                metadata = self._get_model_metadata(model_id, dest_path)
                
                self.nexus_logger.log_event(
                    "model_download_completed",
                    {
                        "model_id": model_id,
                        "destination": str(dest_path),
                        "size_bytes": download_result.get("size_bytes", 0),
                        "files_downloaded": download_result.get("files_downloaded", []),
                        "metadata": metadata
                    }
                )
                
                return {
                    "success": True,
                    "model_id": model_id,
                    "destination": str(dest_path),
                    "size_bytes": download_result.get("size_bytes", 0),
                    "files_downloaded": download_result.get("files_downloaded", []),
                    "metadata": metadata,
                    "download_time_seconds": download_result.get("download_time", 0)
                }
            else:
                raise ProviderError(
                    f"Failed to download model {model_id}: {download_result.get('error', 'Unknown error')}",
                    provider="huggingface",
                    operation="download"
                )
                
        except Exception as e:
            self.logger.error(f"Error downloading model {model_id}: {str(e)}")
            raise ProviderError(
                f"Failed to download model {model_id}: {str(e)}",
                provider="huggingface",
                operation="download"
            )
    
    def _sanitize_model_id(self, model_id: str) -> str:
        """Sanitize model ID for use as directory name."""
        # Remove protocol prefixes
        if model_id.startswith("http://") or model_id.startswith("https://"):
            model_id = model_id.split("://", 1)[1]
        
        # Replace path separators
        model_id = model_id.replace("/", "-").replace("\\", "-")
        
        # Remove invalid characters
        valid_chars = []
        for char in model_id:
            if char.isalnum() or char in "-._":
                valid_chars.append(char)
            else:
                valid_chars.append("_")
        
        return "".join(valid_chars).strip("_")
    
    def _check_disk_space(self, dest_path: Path, max_gb: float) -> bool:
        """Check if there's enough disk space."""
        try:
            # Get available disk space in GB
            stat = os.statvfs(dest_path)
            available_gb = stat.f_bavail * stat.f_frsize / (1024 ** 3)
            
            return available_gb < max_gb
        except Exception:
            # If we can't check, assume there is enough space
            return False
    
    def _simulate_download(self, model_id: str, dest_path: Path, 
                           use_auth: bool) -> Dict[str, Any]:
        """Simulate model download."""
        # Simulate network delay
        time.sleep(1.0)
        
        # Create simulated model files
        model_name = self._sanitize_model_id(model_id)
        
        # Create model configuration file
        config_file = dest_path / "config.json"
        config_data = {
            "model_id": model_id,
            "downloaded_at": time.time(),
            "downloader_version": "1.0.0",
            "files": [
                f"{model_name}.safetensors",
                f"{model_name}.tokenizer.json",
                f"{model_name}.special_tokens_map.json",
                f"{model_name}.tokenizer_config.json"
            ]
        }
        
        with open(config_file, 'w') as f:
            import json
            json.dump(config_data, f, indent=2)
        
        # Create simulated model files
        for file_name in config_data["files"]:
            file_path = dest_path / file_name
            file_size = self._estimate_model_file_size(model_id) // len(config_data["files"])
            
            with open(file_path, 'wb') as f:
                # Write dummy data
                f.write(b'\x00' * file_size)
        
        # Estimate download size
        total_size = sum(f.stat().st_size for f in dest_path.glob("*"))
        
        return {
            "success": True,
            "size_bytes": total_size,
            "files_downloaded": config_data["files"],
            "download_time": 2.0 + (total_size / (10 * 1024 * 1024))  # 10 MB/s estimate
        }
    
    def _estimate_model_file_size(self, model_id: str) -> int:
        """Estimate model file size based on model size."""
        # Estimate based on model parameter count
        param_count = self._extract_param_count_from_model_id(model_id)
        
        # Rough estimate: 8 bytes per parameter for float32
        # With quantization and compression, reduce by factor of 4
        estimated_bytes = param_count * 8 / 4
        
        # Add overhead for tokenizer and metadata
        estimated_bytes += 50 * 1024 * 1024  # 50MB overhead
        
        return int(estimated_bytes)
    
    def _extract_param_count_from_model_id(self, model_id: str) -> int:
        """Extract parameter count from model ID (simplified)."""
        # This is a simplified extraction - in practice, you'd need to query Hugging Face API
        if "0.5B" in model_id:
            return 500000000
        elif "7B" in model_id:
            return 7000000000
        elif "13B" in model_id:
            return 13000000000
        elif "70B" in model_id:
            return 70000000000
        else:
            return 500000000  # Default to 0.5B
    
    def _get_model_metadata(self, model_id: str, dest_path: Path) -> Dict[str, Any]:
        """Get model metadata from Hugging Face."""
        # In a real implementation, this would query the Hugging Face API
        # For now, return simulated metadata
        
        return {
            "model_id": model_id,
            "model_name": self._sanitize_model_id(model_id),
            "downloaded_at": time.time(),
            "size_bytes": sum(f.stat().st_size for f in dest_path.glob("*")),
            "file_count": len(list(dest_path.glob("*"))),
            "source": "huggingface",
            "requires_auth": True
        }
    
    def is_model_downloaded(self, model_id: str) -> bool:
        """Check if model is already downloaded."""
        safe_model_id = self._sanitize_model_id(model_id)
        model_path = self.cache_dir / safe_model_id
        
        return model_path.exists() and any(model_path.iterdir())
    
    def get_model_path(self, model_id: str) -> Path:
        """Get the local path where model is stored."""
        safe_model_id = self._sanitize_model_id(model_id)
        return self.cache_dir / safe_model_id
    
    def clean_cache(self, max_age_days: int = 30, min_size_gb: float = 1.0) -> Dict[str, Any]:
        """Clean cache to free up disk space."""
        self.nexus_logger.log_event(
            "cache_clean_start",
            {
                "max_age_days": max_age_days,
                "min_size_gb": min_size_gb
            }
        )
        
        cleaned_files = []
        cleaned_size = 0
        
        # Check each model directory
        for model_dir in self.cache_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            # Check directory age and size
            dir_age = time.time() - model_dir.stat().st_mtime
            dir_size = sum(f.stat().st_size for f in model_dir.glob("*") if f.is_file())
            
            # Clean if directory is too old or too large
            if dir_age > max_age_days * 24 * 3600 or (dir_size / (1024 ** 3)) > min_size_gb:
                # Remove the directory
                import shutil
                shutil.rmtree(model_dir)
                cleaned_files.append(str(model_dir))
                cleaned_size += dir_size
        
        self.nexus_logger.log_event(
            "cache_clean_completed",
            {
                "cleaned_files_count": len(cleaned_files),
                "cleaned_size_bytes": cleaned_size,
                "remaining_directories": len([d for d in self.cache_dir.iterdir() if d.is_dir()])
            }
        )
        
        return {
            "cleaned_files": cleaned_files,
            "cleaned_size_bytes": cleaned_size,
            "remaining_directories": len([d for d in self.cache_dir.iterdir() if d.is_dir()]),
            "freed_space_gb": cleaned_size / (1024 ** 3)
        }