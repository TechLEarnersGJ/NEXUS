"""
Dataset registry for NEXUS.

Manages the registry of available datasets, including local and remote datasets,
with capabilities for searching, filtering, and managing dataset metadata.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from nexus.core.logger import get_logger
from nexus.datasets.discovery import DatasetMetadata


class DatasetRegistry:
    """Registry for dataset metadata."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "workspace/datasets_registry")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._cache: Dict[str, DatasetMetadata] = {}
        
        # Registry file
        self._registry_file = self.storage_path / "dataset_registry.json"
        
        # Load registry
        self._load_registry()
    
    def _load_registry(self):
        """Load dataset registry from disk."""
        if self._registry_file.exists():
            try:
                with open(self._registry_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to DatasetMetadata objects
                for dataset_id, dataset_data in data.items():
                    try:
                        self._cache[dataset_id] = DatasetMetadata.from_dict(dataset_data)
                    except Exception as e:
                        print(f"Warning: Failed to load dataset {dataset_id}: {str(e)}")
                        
            except Exception as e:
                print(f"Warning: Failed to load dataset registry: {str(e)}")
    
    def save_dataset(self, dataset: DatasetMetadata):
        """Save dataset metadata to registry."""
        self._cache[dataset.dataset_id] = dataset
        
        # Save to disk
        try:
            with open(self._registry_file, 'w') as f:
                data = {dataset_id: dataset.to_dict() for dataset_id, dataset in self._cache.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save dataset registry: {str(e)}")
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get dataset by ID."""
        return self._cache.get(dataset_id)
    
    def remove_dataset(self, dataset_id: str):
        """Remove dataset from registry."""
        if dataset_id in self._cache:
            del self._cache[dataset_id]
        
        # Save to disk
        try:
            with open(self._registry_file, 'w') as f:
                data = {dataset_id: dataset.to_dict() for dataset_id, dataset in self._cache.items()}
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def list_datasets(self, language: Optional[str] = None, 
                      license: Optional[str] = None,
                      task_type: Optional[str] = None) -> List[DatasetMetadata]:
        """List datasets with optional filtering."""
        datasets = list(self._cache.values())
        
        if language:
            datasets = [d for d in datasets if language in d.languages]
        
        if license:
            datasets = [d for d in datasets if d.license and d.license.lower() == license.lower()]
        
        if task_type:
            datasets = [d for d in datasets if d.task_type == task_type]
        
        return datasets
    
    def search_datasets(self, query: str, max_results: int = 10) -> List[DatasetMetadata]:
        """Search datasets by query."""
        datasets = list(self._cache.values())
        
        # Calculate relevance scores
        scored_datasets = []
        for dataset in datasets:
            score = dataset.get_relevance_score(query)
            scored_datasets.append((dataset, score))
        
        # Sort by score (descending)
        scored_datasets.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        return [dataset for dataset, _ in scored_datasets[:max_results]]
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed dataset information."""
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None
        
        return {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "size": dataset.size,
            "size_gb": dataset.get_size_gb(),
            "license": dataset.license,
            "tags": dataset.tags,
            "languages": dataset.languages,
            "usecases": dataset.usecases,
            "task_type": dataset.task_type,
            "downloads": dataset.downloads,
            "likes": dataset.likes,
            "author": dataset.author,
            "created_at": dataset.created_at,
            "last_updated": dataset.last_updated,
            "verification_status": dataset.verification_status
        }