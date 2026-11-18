"""
Dataset formatter for NEXUS.

Formats datasets into the required format for training, including
tokenization, format conversion, and train/validation splits.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


class DatasetFormatter:
    """Formats datasets for training."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.config = config or {}
        
        # Formatting settings
        self.format_type = self.config.get("format_type", "jsonl")
        self.train_split = self.config.get("train_split", 0.8)
        self.validation_split = self.config.get("validation_split", 0.1)
        self.test_split = self.config.get("test_split", 0.1)
        self.shuffle = self.config.get("shuffle", True)
        self.seed = self.config.get("seed", 42)
        
        # Text processing
        self.max_length = self.config.get("max_length", 2048)
        self.truncate = self.config.get("truncate", True)
    
    def format_dataset(self, dataset_path: str, output_path: Optional[str] = None,
                      format_type: Optional[str] = None) -> Dict[str, Any]:
        """Format a dataset for training."""
        self.nexus_logger.log_event(
            "dataset_formatting_start",
            {
                "dataset_path": dataset_path,
                "output_path": output_path,
                "format_type": format_type or self.format_type
            }
        )
        
        try:
            dataset_path = Path(dataset_path)
            output_path = Path(output_path) if output_path else dataset_path / "formatted"
            output_path.mkdir(parents=True, exist_ok=True)
            
            format_type = format_type or self.format_type
            
            # Process dataset files
            formatted_files = []
            all_samples = []
            
            for file in dataset_path.glob("*"):
                if not file.is_file() or file.suffix not in [".json", ".jsonl", ".csv"]:
                    continue
                
                samples = self._read_samples(file)
                all_samples.extend(samples)
            
            # Shuffle if requested
            if self.shuffle:
                import random
                random.seed(self.seed)
                random.shuffle(all_samples)
            
            # Split dataset
            train_samples, val_samples, test_samples = self._split_dataset(all_samples)
            
            # Format and save each split
            splits = {
                "train": (train_samples, "train.jsonl"),
                "validation": (val_samples, "validation.jsonl"),
                "test": (test_samples, "test.jsonl")
            }
            
            for split_name, (samples, filename) in splits.items():
                if not samples:
                    continue
                
                output_file = output_path / filename
                self._save_formatted_samples(samples, output_file, format_type)
                
                formatted_files.append({
                    "split": split_name,
                    "filename": filename,
                    "count": len(samples),
                    "path": str(output_file)
                })
            
            # Create dataset info
            dataset_info = self._create_dataset_info(all_samples, formatted_files)
            
            with open(output_path / "dataset_info.json", 'w') as f:
                import json
                json.dump(dataset_info, f, indent=2)
            
            stats = {
                "formatted_files": formatted_files,
                "total_samples": len(all_samples),
                "train_count": len(train_samples),
                "validation_count": len(val_samples),
                "test_count": len(test_samples),
                "format_type": format_type,
                "output_path": str(output_path)
            }
            
            self.nexus_logger.log_event(
                "dataset_formatting_completed",
                {
                    "dataset_path": str(dataset_path),
                    "output_path": str(output_path),
                    "total_samples": len(all_samples),
                    "format_type": format_type
                }
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Dataset formatting failed: {str(e)}")
            raise
    
    def _read_samples(self, file: Path) -> List[Dict[str, Any]]:
        """Read samples from a file."""
        samples = []
        
        try:
            if file.suffix == ".jsonl":
                with open(file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                samples.append(json.loads(line))
                            except json.JSONDecodeError:
                                self.logger.warning(f"Invalid JSON in {file}: {line}")
                                continue
            
            elif file.suffix == ".json":
                with open(file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples.extend(data)
                    elif isinstance(data, dict):
                        samples.append(data)
            
            elif file.suffix == ".csv":
                import csv
                with open(file, 'r') as f:
                    reader = csv.DictReader(f)
                    samples.extend(list(reader))
        
        except Exception as e:
            self.logger.warning(f"Could not read {file}: {str(e)}")
        
        return samples
    
    def _split_dataset(self, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset into train/validation/test sets."""
        total = len(samples)
        
        train_count = int(total * self.train_split)
        val_count = int(total * self.validation_split)
        test_count = total - train_count - val_count
        
        train_samples = samples[:train_count]
        val_samples = samples[train_count:train_count + val_count]
        test_samples = samples[train_count + val_count:]
        
        return train_samples, val_samples, test_samples
    
    def _save_formatted_samples(self, samples: List[Dict[str, Any]], output_file: Path, format_type: str):
        """Save formatted samples to file."""
        if format_type == "jsonl":
            with open(output_file, 'w') as f:
                for sample in samples:
                    formatted_sample = self._format_sample(sample, format_type)
                    f.write(json.dumps(formatted_sample) + "\n")
        
        elif format_type == "json":
            formatted_samples = [self._format_sample(s, format_type) for s in samples]
            with open(output_file, 'w') as f:
                json.dump(formatted_samples, f, indent=2)
        
        elif format_type == "csv":
            import csv
            if samples:
                fieldnames = samples[0].keys()
                with open(output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for sample in samples:
                        formatted_sample = self._format_sample(sample, format_type)
                        writer.writerow(formatted_sample)
    
    def _format_sample(self, sample: Dict[str, Any], format_type: str) -> Dict[str, Any]:
        """Format a single sample."""
        # Create a standardized format
        formatted = {}
        
        # Handle common field names
        text_fields = ["text", "content", "prompt", "input", "query"]
        for field in text_fields:
            if field in sample and sample[field]:
                formatted["text"] = str(sample[field])[:self.max_length] if self.truncate else str(sample[field])
                break
        
        # Handle label/response fields
        label_fields = ["label", "response", "answer", "output", "completion"]
        for field in label_fields:
            if field in sample and sample[field]:
                formatted["response"] = str(sample[field])[:self.max_length] if self.truncate else str(sample[field])
                break
        
        # Handle instruction fields
        instruction_fields = ["instruction", "task", "question"]
        for field in instruction_fields:
            if field in sample and sample[field]:
                formatted["instruction"] = str(sample[field])[:self.max_length] if self.truncate else str(sample[field])
                break
        
        # Copy other fields
        for key, value in sample.items():
            if key not in formatted and key not in text_fields + label_fields + instruction_fields:
                formatted[key] = value
        
        # Ensure required fields exist
        if "text" not in formatted and "instruction" in formatted:
            formatted["text"] = formatted["instruction"]
        
        if "response" not in formatted:
            formatted["response"] = ""
        
        return formatted
    
    def _create_dataset_info(self, samples: List[Dict[str, Any]], formatted_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create dataset information."""
        return {
            "total_samples": len(samples),
            "splits": {
                file["split"]: {
                    "count": file["count"],
                    "filename": file["filename"]
                } for file in formatted_files
            },
            "format_type": self.format_type,
            "max_length": self.max_length,
            "created_at": json.dumps({"timestamp": "now"}, default=str),
            "features": list(samples[0].keys()) if samples else []
        }