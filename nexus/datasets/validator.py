"""
Dataset validator for NEXUS.

Validates datasets for quality, format, licensing, and usability
before training.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from nexus.core.exceptions import ValidationError
from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


class DatasetValidator:
    """Validates datasets for quality and usability."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.config = config or {}
        
        # Validation settings
        self.require_license = self.config.get("require_license", True)
        self.min_samples = self.config.get("min_samples", 100)
        self.max_samples = self.config.get("max_samples", 1000000)
        self.allowed_licenses = self.config.get("allowed_licenses", 
            ["MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "CC-BY-4.0", "CC-BY-SA-4.0"])
        self.required_columns = self.config.get("required_columns", ["text"])
    
    def validate_dataset(self, dataset_path: str, dataset_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate a dataset."""
        self.nexus_logger.log_event(
            "dataset_validation_start",
            {
                "dataset_path": dataset_path,
                "dataset_metadata": dataset_metadata or {}
            }
        )
        
        try:
            dataset_path = Path(dataset_path)
            
            if not dataset_path.exists():
                raise ValidationError(f"Dataset path does not exist: {dataset_path}")
            
            # Perform various validation checks
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "checks": {}
            }
            
            # Check file existence and format
            file_check = self._check_file_format(dataset_path)
            validation_results["checks"]["file_format"] = file_check
            if not file_check["valid"]:
                validation_results["errors"].extend(file_check["errors"])
                validation_results["valid"] = False
            
            # Check dataset size
            size_check = self._check_dataset_size(dataset_path)
            validation_results["checks"]["dataset_size"] = size_check
            if not size_check["valid"]:
                validation_results["errors"].extend(size_check["errors"])
                validation_results["valid"] = False
            elif size_check["warnings"]:
                validation_results["warnings"].extend(size_check["warnings"])
            
            # Check license
            if self.require_license:
                license_check = self._check_license(dataset_metadata)
                validation_results["checks"]["license"] = license_check
                if not license_check["valid"]:
                    validation_results["errors"].extend(license_check["errors"])
                    validation_results["valid"] = False
                elif license_check["warnings"]:
                    validation_results["warnings"].extend(license_check["warnings"])
            
            # Check required columns
            column_check = self._check_required_columns(dataset_path)
            validation_results["checks"]["required_columns"] = column_check
            if not column_check["valid"]:
                validation_results["errors"].extend(column_check["errors"])
                validation_results["valid"] = False
            elif column_check["warnings"]:
                validation_results["warnings"].extend(column_check["warnings"])
            
            # Check data quality
            quality_check = self._check_data_quality(dataset_path)
            validation_results["checks"]["data_quality"] = quality_check
            if not quality_check["valid"]:
                validation_results["errors"].extend(quality_check["errors"])
                validation_results["valid"] = False
            elif quality_check["warnings"]:
                validation_results["warnings"].extend(quality_check["warnings"])
            
            self.nexus_logger.log_event(
                "dataset_validation_completed",
                {
                    "dataset_path": str(dataset_path),
                    "valid": validation_results["valid"],
                    "errors_count": len(validation_results["errors"]),
                    "warnings_count": len(validation_results["warnings"])
                }
            )
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {str(e)}")
            raise ValidationError(f"Dataset validation failed: {str(e)}")
    
    def _check_file_format(self, dataset_path: Path) -> Dict[str, Any]:
        """Check dataset file format."""
        result = {"valid": True, "errors": [], "warnings": [], "details": {}}
        
        # Check for supported file formats
        supported_extensions = [".json", ".jsonl", ".csv", ".parquet", ".arrow", ".txt"]
        
        files = []
        for ext in supported_extensions:
            files.extend(dataset_path.glob(f"*{ext}"))
        
        if not files:
            result["valid"] = False
            result["errors"].append(f"No supported dataset files found in {dataset_path}")
            return result
        
        result["details"]["file_count"] = len(files)
        result["details"]["file_types"] = list(set(f.suffix for f in files))
        
        # Check for metadata file
        metadata_files = list(dataset_path.glob("*.json"))
        if not metadata_files:
            result["warnings"].append("No metadata file found")
        
        return result
    
    def _check_dataset_size(self, dataset_path: Path) -> Dict[str, Any]:
        """Check dataset size."""
        result = {"valid": True, "errors": [], "warnings": [], "details": {}}
        
        # Count total rows (simplified)
        total_rows = 0
        for file in dataset_path.glob("*"):
            if file.is_file():
                if file.suffix == ".jsonl":
                    with open(file, 'r') as f:
                        total_rows += sum(1 for _ in f)
                elif file.suffix == ".csv":
                    with open(file, 'r') as f:
                        total_rows += sum(1 for _ in f) - 1  # Subtract header
                elif file.suffix in [".json", ".parquet", ".arrow"]:
                    # Estimate based on file size
                    file_size = file.stat().st_size
                    total_rows += file_size // 1000  # Rough estimate
        
        result["details"]["total_rows"] = total_rows
        
        if total_rows < self.min_samples:
            result["valid"] = False
            result["errors"].append(f"Dataset has {total_rows} rows, minimum required is {self.min_samples}")
        elif total_rows > self.max_samples:
            result["valid"] = False
            result["errors"].append(f"Dataset has {total_rows} rows, maximum allowed is {self.max_samples}")
        elif total_rows < self.min_samples * 2:
            result["warnings"].append(f"Dataset has {total_rows} rows, which is below recommended minimum of {self.min_samples * 2}")
        
        return result
    
    def _check_license(self, dataset_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check dataset license."""
        result = {"valid": True, "errors": [], "warnings": [], "details": {}}
        
        if not dataset_metadata:
            result["warnings"].append("No dataset metadata provided, cannot verify license")
            return result
        
        license_info = dataset_metadata.get("license", "")
        result["details"]["license"] = license_info
        
        if not license_info:
            result["valid"] = False
            result["errors"].append("Dataset license not specified")
            return result
        
        if license_info not in self.allowed_licenses:
            result["valid"] = False
            result["errors"].append(f"Dataset license '{license_info}' is not in allowed licenses: {self.allowed_licenses}")
        else:
            result["details"]["license_allowed"] = True
        
        return result
    
    def _check_required_columns(self, dataset_path: Path) -> Dict[str, Any]:
        """Check for required columns in dataset."""
        result = {"valid": True, "errors": [], "warnings": [], "details": {}}
        
        # Try to infer columns from dataset files
        found_columns = []
        
        for file in dataset_path.glob("*"):
            if file.is_file() and file.suffix in [".json", ".jsonl", ".csv"]:
                try:
                    if file.suffix == ".json":
                        with open(file, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                found_columns.extend(data.keys())
                            elif isinstance(data, list) and data:
                                if isinstance(data[0], dict):
                                    found_columns.extend(data[0].keys())
                    
                    elif file.suffix == ".jsonl":
                        with open(file, 'r') as f:
                            first_line = f.readline()
                            if first_line:
                                data = json.loads(first_line)
                                if isinstance(data, dict):
                                    found_columns.extend(data.keys())
                    
                    elif file.suffix == ".csv":
                        with open(file, 'r') as f:
                            header = f.readline().strip()
                            if header:
                                found_columns.extend(header.split(','))
                    
                except Exception as e:
                    result["warnings"].append(f"Could not parse {file.name}: {str(e)}")
        
        # Remove duplicates
        found_columns = list(set(found_columns))
        result["details"]["found_columns"] = found_columns
        
        # Check required columns
        missing_columns = []
        for required_col in self.required_columns:
            if required_col not in found_columns:
                missing_columns.append(required_col)
        
        if missing_columns:
            result["valid"] = False
            result["errors"].append(f"Missing required columns: {missing_columns}")
        else:
            result["details"]["required_columns_present"] = True
        
        return result
    
    def _check_data_quality(self, dataset_path: Path) -> Dict[str, Any]:
        """Check data quality."""
        result = {"valid": True, "errors": [], "warnings": [], "details": {}}
        
        # Check for null values, empty strings, etc.
        issues = []
        
        for file in dataset_path.glob("*"):
            if file.is_file() and file.suffix in [".json", ".jsonl", ".csv"]:
                try:
                    # Simple quality checks
                    if file.suffix == ".jsonl":
                        with open(file, 'r') as f:
                            for i, line in enumerate(f, 1):
                                if i > 100:  # Only check first 100 lines
                                    break
                                try:
                                    data = json.loads(line)
                                    if isinstance(data, dict):
                                        # Check for empty values
                                        for key, value in data.items():
                                            if value is None or value == "":
                                                issues.append(f"Empty value in {file.name}, line {i}, field {key}")
                                except json.JSONDecodeError:
                                    issues.append(f"Invalid JSON in {file.name}, line {i}")
                    
                except Exception as e:
                    result["warnings"].append(f"Could not check quality for {file.name}: {str(e)}")
        
        if len(issues) > 10:
            result["warnings"].append(f"Found {len(issues)} data quality issues (showing first 10)")
            result["warnings"].extend(issues[:10])
        elif issues:
            result["warnings"].extend(issues)
        
        result["details"]["quality_issues_count"] = len(issues)
        
        return result