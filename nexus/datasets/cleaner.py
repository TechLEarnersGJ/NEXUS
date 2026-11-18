"""
Dataset cleaner for NEXUS.

Cleans and preprocesses datasets, removing duplicates, invalid samples,
and normalizing data for training.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any

from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


class DatasetCleaner:
    """Cleans and preprocesses datasets."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.config = config or {}
        
        # Cleaning settings
        self.remove_duplicates = self.config.get("remove_duplicates", True)
        self.remove_empty = self.config.get("remove_empty", True)
        self.normalize_text = self.config.get("normalize_text", True)
        self.min_text_length = self.config.get("min_text_length", 10)
        self.max_text_length = self.config.get("max_text_length", 10000)
    
    def clean_dataset(self, dataset_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Clean a dataset."""
        self.nexus_logger.log_event(
            "dataset_cleaning_start",
            {
                "dataset_path": dataset_path,
                "output_path": output_path
            }
        )
        
        try:
            dataset_path = Path(dataset_path)
            output_path = Path(output_path) if output_path else dataset_path / "cleaned"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Clean dataset files
            cleaned_files = []
            total_original = 0
            total_cleaned = 0
            
            for file in dataset_path.glob("*"):
                if not file.is_file() or file.suffix not in [".json", ".jsonl", ".csv"]:
                    continue
                
                if file.suffix == ".jsonl":
                    result = self._clean_jsonl_file(file, output_path)
                elif file.suffix == ".json":
                    result = self._clean_json_file(file, output_path)
                elif file.suffix == ".csv":
                    result = self._clean_csv_file(file, output_path)
                else:
                    continue
                
                cleaned_files.append(result)
                total_original += result["original_count"]
                total_cleaned += result["cleaned_count"]
            
            # Calculate cleaning statistics
            stats = {
                "cleaned_files": cleaned_files,
                "total_original_samples": total_original,
                "total_cleaned_samples": total_cleaned,
                "samples_removed": total_original - total_cleaned,
                "removal_rate": (total_original - total_cleaned) / total_original if total_original > 0 else 0,
                "output_path": str(output_path)
            }
            
            self.nexus_logger.log_event(
                "dataset_cleaning_completed",
                {
                    "dataset_path": str(dataset_path),
                    "output_path": str(output_path),
                    "total_original": total_original,
                    "total_cleaned": total_cleaned,
                    "samples_removed": total_original - total_cleaned
                }
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Dataset cleaning failed: {str(e)}")
            raise
    
    def _clean_jsonl_file(self, input_file: Path, output_path: Path) -> Dict[str, Any]:
        """Clean a JSONL file."""
        original_count = 0
        cleaned_count = 0
        seen_hashes = set()
        
        output_file = output_path / f"cleaned_{input_file.name}"
        
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                original_count += 1
                
                try:
                    data = json.loads(line.strip())
                    
                    # Clean the data
                    cleaned_data = self._clean_sample(data)
                    
                    if cleaned_data is None:
                        continue
                    
                    # Check for duplicates
                    if self.remove_duplicates:
                        sample_hash = self._get_sample_hash(cleaned_data)
                        if sample_hash in seen_hashes:
                            continue
                        seen_hashes.add(sample_hash)
                    
                    # Write cleaned data
                    f_out.write(json.dumps(cleaned_data) + "\n")
                    cleaned_count += 1
                    
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON in {input_file}: {line.strip()}")
                    continue
        
        return {
            "file": str(input_file),
            "output_file": str(output_file),
            "original_count": original_count,
            "cleaned_count": cleaned_count,
            "removed_count": original_count - cleaned_count
        }
    
    def _clean_json_file(self, input_file: Path, output_path: Path) -> Dict[str, Any]:
        """Clean a JSON file."""
        original_count = 0
        cleaned_count = 0
        seen_hashes = set()
        
        output_file = output_path / f"cleaned_{input_file.name}"
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            cleaned_data = []
            
            for item in data:
                original_count += 1
                
                # Clean the data
                cleaned_item = self._clean_sample(item)
                
                if cleaned_item is None:
                    continue
                
                # Check for duplicates
                if self.remove_duplicates:
                    sample_hash = self._get_sample_hash(cleaned_item)
                    if sample_hash in seen_hashes:
                        continue
                    seen_hashes.add(sample_hash)
                
                cleaned_data.append(cleaned_item)
                cleaned_count += 1
            
            with open(output_file, 'w') as f:
                json.dump(cleaned_data, f, indent=2)
        
        elif isinstance(data, dict):
            # Handle single object
            original_count = 1
            cleaned_data = self._clean_sample(data)
            
            if cleaned_data is not None:
                cleaned_count = 1
                with open(output_file, 'w') as f:
                    json.dump(cleaned_data, f, indent=2)
        
        return {
            "file": str(input_file),
            "output_file": str(output_file),
            "original_count": original_count,
            "cleaned_count": cleaned_count,
            "removed_count": original_count - cleaned_count
        }
    
    def _clean_csv_file(self, input_file: Path, output_path: Path) -> Dict[str, Any]:
        """Clean a CSV file."""
        import csv
        
        original_count = 0
        cleaned_count = 0
        seen_hashes = set()
        
        output_file = output_path / f"cleaned_{input_file.name}"
        
        with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
            reader = csv.DictReader(f_in)
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()
            
            for row in reader:
                original_count += 1
                
                # Clean the data
                cleaned_row = self._clean_sample(row)
                
                if cleaned_row is None:
                    continue
                
                # Check for duplicates
                if self.remove_duplicates:
                    sample_hash = self._get_sample_hash(cleaned_row)
                    if sample_hash in seen_hashes:
                        continue
                    seen_hashes.add(sample_hash)
                
                writer.writerow(cleaned_row)
                cleaned_count += 1
        
        return {
            "file": str(input_file),
            "output_file": str(output_file),
            "original_count": original_count,
            "cleaned_count": cleaned_count,
            "removed_count": original_count - cleaned_count
        }
    
    def _clean_sample(self, sample: Any) -> Optional[Dict[str, Any]]:
        """Clean a single sample."""
        if not isinstance(sample, dict):
            return None
        
        # Remove empty samples
        if self.remove_empty:
            if all(value is None or value == "" for value in sample.values()):
                return None
        
        # Clean text fields
        cleaned_sample = {}
        for key, value in sample.items():
            if isinstance(value, str):
                cleaned_value = self._clean_text(value)
                if cleaned_value is None:
                    return None
                cleaned_sample[key] = cleaned_value
            else:
                cleaned_sample[key] = value
        
        return cleaned_sample
    
    def _clean_text(self, text: str) -> Optional[str]:
        """Clean a text field."""
        if not isinstance(text, str):
            return text
        
        # Normalize text
        if self.normalize_text:
            # Remove extra whitespace
            text = " ".join(text.split())
            
            # Remove leading/trailing whitespace
            text = text.strip()
        
        # Check text length
        if len(text) < self.min_text_length:
            return None
        
        if len(text) > self.max_text_length:
            return None
        
        return text
    
    def _get_sample_hash(self, sample: Dict[str, Any]) -> str:
        """Get a hash for a sample (for duplicate detection)."""
        # Create a canonical representation
        canonical = json.dumps(sample, sort_keys=True)
        return hashlib.md5(canonical.encode()).hexdigest()