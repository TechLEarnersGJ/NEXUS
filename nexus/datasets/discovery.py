"""
Dataset discovery for NEXUS.

Provides functionality to discover and search for datasets on Hugging Face,
collecting metadata, filtering by criteria, and ranking by relevance.
"""

import time
from typing import Dict, List, Optional, Any

from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger
from nexus.models.metadata import ModelMetadata


class DatasetDiscovery:
    """Discovers and searches for datasets on Hugging Face."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self._datasets_cache: Dict[str, Dict[str, Any]] = {}
        self._last_update: float = 0.0
    
    def search(self, query: Optional[str] = None, max_datasets: int = 10,
              filters: Optional[Dict[str, Any]] = None) -> List["DatasetMetadata"]:
        """Search for datasets."""
        self.nexus_logger.log_event(
            "dataset_discovery_search",
            {
                "query": query,
                "max_datasets": max_datasets,
                "filters": filters or {}
            }
        )
        
        # Simulate dataset discovery
        datasets = self._fetch_datasets_from_huggingface(query, max_datasets, filters)
        
        # Convert to DatasetMetadata objects
        dataset_objects = []
        for dataset_data in datasets:
            try:
                dataset_objects.append(DatasetMetadata.from_dict(dataset_data))
            except Exception as e:
                self.logger.warning(f"Failed to parse dataset metadata: {str(e)}")
                continue
        
        self.nexus_logger.log_event(
            "dataset_discovery_search_completed",
            {
                "query": query,
                "max_datasets": max_datasets,
                "filters": filters or {},
                "datasets_found": len(dataset_objects)
            }
        )
        
        return dataset_objects
    
    def _fetch_datasets_from_huggingface(self, query: Optional[str], 
                                          max_datasets: int,
                                          filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch datasets from Hugging Face API."""
        # Simulate API response
        simulated_datasets = []
        
        # Popular datasets with various characteristics
        dataset_templates = [
            {
                "id": "iamtarun/python_code_instructions_18k_alpaca",
                "dataset_id": "python-code-instructions-18k-alpaca",
                "name": "Python Code Instructions 18K Alpaca",
                "description": "Python code instructions dataset for training code generation models",
                "size": 4069935,
                "license": "Apache-2.0",
                "tags": ["python", "code", "instructions", "programming"],
                "downloads": 2401,
                "likes": 35,
                "usecases": ["code-generation", "instruction-following"],
                "languages": ["python", "javascript"],
                "format": "dataset",
                "task_type": "code-generation"
            },
            {
                "id": "ipython/python-code-dataset",
                "dataset_id": "python-code-dataset",
                "name": "Python Code Dataset",
                "description": "Large dataset of Python code snippets",
                "size": 62220335,
                "license": "Apache-2.0",
                "tags": ["python", "code", "programming", "large-dataset"],
                "downloads": 1483,
                "likes": 27,
                "usecases": ["code-generation", "code-analysis"],
                "languages": ["python"],
                "format": "dataset",
                "task_type": "code-generation"
            },
            {
                "id": "veeralakrishna/150k-python-dataset",
                "dataset_id": "150k-python-dataset",
                "name": "150K Python Dataset",
                "description": "Large collection of Python code examples",
                "size": 532039302,
                "license": "CC0",
                "tags": ["python", "code", "large-dataset", "examples"],
                "downloads": 3619,
                "likes": 68,
                "usecases": ["code-generation", "pattern-recognition"],
                "languages": ["python"],
                "format": "dataset",
                "task_type": "code-generation"
            },
            {
                "id": "linkanjarad/coding-problems-and-solution-python-code",
                "dataset_id": "coding-problems-and-solution-python-code",
                "name": "Coding Problems and Solutions",
                "description": "Python coding problems with solutions",
                "size": 1481269,
                "license": "MIT",
                "tags": ["python", "coding-problems", "solutions", "educational"],
                "downloads": 1751,
                "likes": 31,
                "usecases": ["code-generation", "problem-solving"],
                "languages": ["python"],
                "format": "dataset",
                "task_type": "code-generation"
            },
            {
                "id": "shamimhasan8/python-code-bug-and-fix-pairs",
                "dataset_id": "python-code-bug-and-fix-pairs",
                "name": "Python Code Bug & Fix Pairs",
                "description": "Python code with bugs and their fixes",
                "size": 42551,
                "license": "Apache-2.0",
                "tags": ["python", "debugging", "bugs", "fixes"],
                "downloads": 773,
                "likes": 17,
                "usecases": ["bug-fix-detection", "code-correction"],
                "languages": ["python"],
                "format": "dataset",
                "task_type": "bug-detection"
            }
        ]
        
        # Filter based on query
        if query:
            query_lower = query.lower()
            dataset_templates = [
                d for d in dataset_templates
                if query_lower in d["name"].lower()
                or query_lower in d["description"].lower()
                or any(query_lower in tag.lower() for tag in d["tags"])
            ]
        
        # Apply filters
        if filters:
            if "license" in filters:
                dataset_templates = [d for d in dataset_templates 
                                   if d.get("license", "").lower() == filters["license"].lower()]
            
            if "size" in filters:
                dataset_templates = [d for d in dataset_templates 
                                   if d.get("size", 0) <= filters["size"]]
            
            if "tags" in filters:
                dataset_templates = [d for d in dataset_templates 
                                   if any(tag in filters["tags"] for tag in d.get("tags", []))]
            
            if "languages" in filters:
                dataset_templates = [d for d in dataset_templates 
                                   if any(lang in filters["languages"] for lang in d.get("languages", []))]
        
        # Sort by downloads/likes score
        dataset_templates.sort(key=lambda d: (d.get("downloads", 0) + d.get("likes", 0)) / 100, reverse=True)
        
        # Return top N datasets
        return dataset_templates[:max_datasets]


class DatasetMetadata:
    """Dataset metadata information."""
    
    def __init__(self, dataset_data: Dict[str, Any]):
        self.id = dataset_data.get("id", "")
        self.dataset_id = dataset_data.get("dataset_id", self.id)
        self.name = dataset_data.get("name", "")
        self.description = dataset_data.get("description", "")
        self.size = dataset_data.get("size", 0)
        self.license = dataset_data.get("license", "")
        self.tags = dataset_data.get("tags", [])
        self.downloads = dataset_data.get("downloads", 0)
        self.likes = dataset_data.get("likes", 0)
        self.usecases = dataset_data.get("usecases", [])
        self.languages = dataset_data.get("languages", [])
        self.format = dataset_data.get("format", "")
        self.task_type = dataset_data.get("task_type", "")
        self.created_at = time.time()
        self.last_updated = time.time()
        self.author = dataset_data.get("author", "")
        self.paper_url = dataset_data.get("paper_url", "")
        self.dataset_url = dataset_data.get("dataset_url", "")
        self.verification_status = dataset_data.get("verification_status", "unverified")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        """Create DatasetMetadata from dictionary."""
        return cls(data)
    
    @classmethod
    def from_huggingface_api(cls, api_data: Dict[str, Any]) -> "DatasetMetadata":
        """Create DatasetMetadata from Hugging Face API response."""
        # Map Hugging Face API fields to our metadata
        metadata = {
            "id": api_data.get("id", ""),
            "dataset_id": api_data.get("datasetId", api_data.get("id", "")),
            "name": api_data.get("name", ""),
            "description": api_data.get("description", ""),
            "size": api_data.get("size", 0),
            "license": api_data.get("license", ""),
            "tags": api_data.get("tags", []),
            "downloads": api_data.get("downloads", 0),
            "likes": api_data.get("likes", 0),
            "usecases": api_data.get("usecases", []),
            "languages": api_data.get("languages", []),
            "format": api_data.get("format", ""),
            "task_type": api_data.get("taskType", api_data.get("task_type", "")),
            "created_at": time.time(),
            "last_updated": api_data.get("lastModified", time.time()),
            "author": api_data.get("author", ""),
            "paper_url": api_data.get("paperUrl", ""),
            "dataset_url": f"https://huggingface.co/datasets/{api_data.get('id', '')}",
            "verification_status": api_data.get("verificationStatus", "unverified")
        }
        
        return cls(**metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DatasetMetadata to dictionary."""
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "size": self.size,
            "license": self.license,
            "tags": self.tags,
            "downloads": self.downloads,
            "likes": self.likes,
            "usecases": self.usecases,
            "languages": self.languages,
            "format": self.format,
            "task_type": self.task_type,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "author": self.author,
            "paper_url": self.paper_url,
            "dataset_url": self.dataset_url,
            "verification_status": self.verification_status
        }
    
    def get_size_gb(self) -> float:
        """Get dataset size in GB."""
        return self.size / (1024 ** 3)
    
    def is_large_dataset(self, threshold_gb: float = 10.0) -> bool:
        """Check if dataset is large (more than threshold GB)."""
        return self.size > threshold_gb * (1024 ** 3)
    
    def get_relevance_score(self, query: str) -> float:
        """Calculate relevance score for a query."""
        score = 0.0
        query_lower = query.lower()
        
        # Check for exact name match
        if query_lower == self.name.lower():
            score += 10.0
        
        # Check for tag matches
        tag_matches = sum(1 for tag in self.tags if query_lower in tag.lower())
        score += tag_matches * 3.0
        
        # Check for description matches
        if query_lower in self.description.lower():
            score += 5.0
        
        # Check for language relevance
        language_matches = sum(1 for lang in self.languages if query_lower == lang.lower())
        score += language_matches * 4.0
        
        # Check for use case matches
        usecase_matches = sum(1 for usecase in self.usecases if query_lower in usecase.lower())
        score += usecase_matches * 2.0
        
        # Normalize score
        score = min(score, 10.0)
        
        return score