"""
Model validator for NEXUS.

Validates models for compatibility, licensing, and usability before training.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from nexus.core.exceptions import ValidationError
from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


class ModelValidator:
    """Validates models for compatibility and usability."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.config = config or {}
        
        # Validation settings
        self.require_license = self.config.get("require_license", True)
        self.allowed_licenses = self.config.get("allowed_licenses", 
            ["MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "CC-BY-4.0", "CC-BY-SA-4.0", "LLAMA2"])
        self.max_model_size_gb = self.config.get("max_model_size_gb", 100)
        self.required_files = self.config.get("required_files", ["config.json"])
    
    def validate_model(self, model_path: str, model_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate a model."""
        self.nexus_logger.log_event(
            "model_validation_start",
            {
                "model_path": model_path,
                "model_metadata": model_metadata or {}
            }
        )
        
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise ValidationError(f"Model path does not exist: {model_path}")
            
            # Perform various validation checks
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "checks": {}
            }
            
            # Check file existence
            file_check = self._check_required_files(model_path)
            validation_results["checks"]["required_files"] = file_check
            if not file_check["valid"]:
                validation_results["errors"].extend(file_check["errors"])
                validation_results["valid"] = False
            
            # Check model size
            size_check = self._check_model_size(model_path)
            validation_results["checks"]["model_size"] = size_check
            if not size_check["valid"]:
                validation_results["errors"].extend(size_check["errors"])
                validation_results["valid"] = False
            elif size_check["warnings"]:
                validation_results["warnings"].extend(size_check["warnings"])
            
            # Check license
            if self.require_license:
                license_check = self._check_license(model_metadata)
                validation_results["checks"]["license"] = license_check
                if not license_check["valid"]:
                    validation_results["errors"].extend(license_check["errors"])
                    validation_results["valid"] = False
                elif license_check["warnings"]:
                    validation_results["warnings"].extend(license_check["warnings"])
            
            # Check model format
            format_check = self._check_model_format(model_path)
            validation_results["checks"]["model_format"] = format_check
            if not format_check["valid"]:
                validation_results["errors"].extend(format_check["errors"])
                validation_results["valid"] = False
            elif format_check["warnings"]:
                validation_results["warnings"].extend(format_check["warnings"])
            
            # Check model compatibility
            compatibility_check = self._check_compatibility(model_path, model_metadata)
            validation_results["checks"]["compatibility"] = compatibility_check
            if not compatibility_check["valid"]:
                validation_results["errors"].extend(compatibility_check["errors"])
                validation_results["valid"] = False
            elif compatibility_check["warnings"]:
                validation_results["warnings"].extend(compatibility_check["warnings"])
            
            self.nexus_logger.log_event(
                "model_validation_completed",
                {
                    "model_path": str(model_path),
                    "valid": validation_results["valid"],
                    "errors_count": len(validation_results["errors"]),
                    "warnings_count": len(validation_results["warnings"])
                }
            )
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}")
            raise ValidationError(f"Model validation failed: {str(e)}")
    
    def _check_required_files(self, model_path: Path) -> Dict[str, Any]:
        """Check for required model files."""
        result = {"valid": True, "errors": [], "warnings": [], "details": {}}
        
        missing_files = []
        for required_file in self.required_files:
            file_path = model_path / required_file
            if not file_path.exists():
                missing_files.append(required_file)
        
        if missing_files:
            result["valid"] = False
            result["errors"].append(f"Missing required files: {missing_files}")
        else:
            result["details"]["all_required_files_present"] = True
        
        # Check for model weight files
        weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin")) + list(model_path.glob("*.pt"))
        result["details"]["weight_files"] = [f.name for f in weight_files]
        
        if not weight_files:
            result["warnings"].append("No model weight files found")
        
        return result
    
    def _check_model_size(self, model_path: Path) -> Dict[str, Any]:
        """Check model size."""
        result = {"valid": True, "errors": [], "warnings": [], "details": {}}
        
        # Calculate total size
        total_size = 0
        for file in model_path.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
        
        size_gb = total_size / (1024 ** 3)
        result["details"]["total_size_bytes"] = total_size
        result["details"]["size_gb"] = size_gb
        
        if size_gb > self.max_model_size_gb:
            result["valid"] = False
            result["errors"].append(f"Model size {size_gb:.2f}GB exceeds maximum {self.max_model_size_gb}GB")
        elif size_gb > self.max_model_size_gb * 0.8:
            result["warnings"].append(f"Model size {size_gb:.2f}GB is close to maximum {self.max_model_size_gb}GB")
        
        return result
    
    def _check_license(self, model_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check model license."""
        result = {"valid": True, "errors": [], "warnings": [], "details": {}}
        
        if not model_metadata:
            result["warnings"].append("No model metadata provided, cannot verify license")
            return result
        
        license_info = model_metadata.get("license", "")
        result["details"]["license"] = license_info
        
        if not license_info:
            result["valid"] = False
            result["errors"].append("Model license not specified")
            return result
        
        if license_info not in self.allowed_licenses:
            result["valid"] = False
            result["errors"].append(f"Model license '{license_info}' is not in allowed licenses: {self.allowed_licenses}")
        else:
            result["details"]["license_allowed"] = True
        
        return result
    
    def _check_model_format(self, model_path: Path) -> Dict[str, Any]:
        """Check model format."""
        result = {"valid": True, "errors": [], "warnings": [], "details": {}}
        
        # Check for common model formats
        formats_found = []
        
        if (model_path / "config.json").exists():
            formats_found.append("transformers")
        
        if list(model_path.glob("*.safetensors")):
            formats_found.append("safetensors")
        
        if list(model_path.glob("*.bin")):
            formats_found.append("pytorch")
        
        if list(model_path.glob("*.pt")):
            formats_found.append("pytorch")
        
        result["details"]["formats_found"] = formats_found
        
        if not formats_found:
            result["valid"] = False
            result["errors"].append("No recognized model format found")
        elif "transformers" not in formats_found:
            result["warnings"].append("Model does not appear to be in Transformers format")
        
        return result
    
    def _check_compatibility(self, model_path: Path, model_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check model compatibility for training."""
        result = {"valid": True, "errors": [], "warnings": [], "details": {}}
        
        # Check if model is compatible with training
        compatibility_issues = []
        
        # Check for config.json
        config_file = model_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    import json
                    config = json.load(f)
                
                # Check model type
                model_type = config.get("model_type", "")
                result["details"]["model_type"] = model_type
                
                # Check for supported model types
                supported_types = ["llama", "qwen", "gemma", "mistral", "phi", "gpt2"]
                if model_type and model_type.lower() not in supported_types:
                    compatibility_issues.append(f"Model type '{model_type}' may not be fully supported")
                
                # Check for LoRA compatibility
                if "lora" in config.get("architectures", []):
                    result["details"]["lora_compatible"] = True
                else:
                    result["details"]["lora_compatible"] = False
                    compatibility_issues.append("Model may not be compatible with LoRA training")
                
            except Exception as e:
                compatibility_issues.append(f"Could not read model config: {str(e)}")
        
        if compatibility_issues:
            result["warnings"].extend(compatibility_issues)
        
        result["details"]["compatibility_issues"] = compatibility_issues
        
        return result