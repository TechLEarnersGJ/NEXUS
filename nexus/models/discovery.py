"""
Model discovery for NEXUS.

Provides functionality to discover and search for models on Hugging Face,
collecting metadata, filtering by criteria, and ranking by relevance.
"""

import time
from typing import Dict, List, Optional, Any

from nexus.core.exceptions import ValidationError
from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


class ModelDiscovery:
    """Discovers and searches for models on Hugging Face."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self._models_cache: Dict[str, Dict[str, Any]] = {}
        self._last_update: float = 0.0
    
    def search(self, query: Optional[str] = None, max_models: int = 10,
              filters: Optional[Dict[str, Any]] = None) -> List["ModelMetadata"]:
        """Search for models."""
        self.nexus_logger.log_event(
            "model_discovery_search",
            {
                "query": query,
                "max_models": max_models,
                "filters": filters or {}
            }
        )
        
        # Simulate model discovery
        models = self._fetch_models_from_huggingface(query, max_models, filters)
        
        # Convert to ModelMetadata objects
        model_objects = []
        for model_data in models:
            try:
                model_objects.append(ModelMetadata.from_dict(model_data))
            except Exception as e:
                self.logger.warning(f"Failed to parse model metadata: {str(e)}")
                continue
        
        self.nexus_logger.log_event(
            "model_discovery_search_completed",
            {
                "query": query,
                "max_models": max_models,
                "filters": filters or {},
                "models_found": len(model_objects)
            }
        )
        
        return model_objects
    
    def _fetch_models_from_huggingface(self, query: Optional[str], 
                                        max_models: int,
                                        filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch models from Hugging Face API."""
        # Simulate API response
        simulated_models = []
        
        # Popular models with various characteristics
        model_templates = [
            {
                "id": f"Qwen/Qwen2.5-0.5B-Instruct",
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "name": "Qwen2.5-0.5B-Instruct",
                "architecture": "Transformers",
                "family": "Qwen",
                "parameter_count": 500000000,
                "size": 2.5,
                "context_length": 8192,
                "license": "Apache-2.0",
                "tags": ["conversational", "instruction-following", "python"],
                "downloads": 1500000,
                "likes": 5000,
                "pipeline_tag": "text-generation",
                "library_name": "transformers",
                "model_size": "2.5GB",
                "quantization": ["4-bit", "8-bit"],
                "finetuned_from": "Qwen2.5-0.5B",
                "dataset_tags": ["python-code", "instruction-dataset"],
                "model_score": 8.5
            },
            {
                "id": "google/gemma-2b-it",
                "model_id": "google/gemma-2b-it",
                "name": "gemma-2b-it",
                "architecture": "AutoModelForCausalLM",
                "family": "Gemma",
                "parameter_count": 2000000000,
                "size": 1.2,
                "context_length": 8192,
                "license": "Apache-2.0",
                "tags": ["instruction-tuned", "conversational", "multilingual"],
                "downloads": 800000,
                "likes": 3000,
                "pipeline_tag": "text-generation",
                "library_name": "transformers",
                "model_size": "1.2GB",
                "quantization": ["4-bit"],
                "finetuned_from": "gemma-2b",
                "dataset_tags": ["instruction-dataset", "conversational"]
            },
            {
                "id": "meta-llama/Llama-2-7b-chat-hf",
                "model_id": "meta-llama/Llama-2-7b-chat-hf",
                "name": "Llama-2-7b-chat-hf",
                "architecture": "AutoModelForCausalLM",
                "family": "Llama",
                "parameter_count": 7000000000,
                "size": 15.0,
                "context_length": 4096,
                "license": "LLAMA2",
                "tags": ["chat", "conversational", "instruction-following"],
                "downloads": 2500000,
                "likes": 8000,
                "pipeline_tag": "text-generation",
                "library_name": "transformers",
                "model_size": "15GB",
                "quantization": ["4-bit", "8-bit"],
                "finetuned_from": "Llama-2-7b",
                "dataset_tags": ["python-code", "instruction-dataset", "chat"]
            },
            {
                "id": "mistralai/Mistral-7B-Instruct-2",
                "model_id": "mistralai/Mistral-7B-Instruct-2",
                "name": "Mistral-7B-Instruct-2",
                "architecture": "AutoModelForCausalLM",
                "family": "Mistral",
                "parameter_count": 7000000000,
                "size": 14.0,
                "context_length": 8192,
                "license": "Apache-2.0",
                "tags": ["instruction-tuned", "conversational", "multilingual"],
                "downloads": 1200000,
                "likes": 4500,
                "pipeline_tag": "text-generation",
                "library_name": "transformers",
                "model_size": "14GB",
                "quantization": ["4-bit"],
                "finetuned_from": "Mistral-7B",
                "dataset_tags": ["instruction-dataset", "conversational", "multilingual"]
            },
            {
                "id": "allenai/OLMo-1B-Instruct",
                "model_id": "allenai/OLMo-1B-Instruct",
                "name": "OLMo-1B-Instruct",
                "architecture": "AutoModelForCausalLM",
                "family": "OLMo",
                "parameter_count": 1000000000,
                "size": 0.8,
                "context_length": 8192,
                "license": "MIT",
                "tags": ["instruction-tuned", "research"],
                "downloads": 300000,
                "likes": 1500,
                "pipeline_tag": "text-generation",
                "library_name": "transformers",
                "model_size": "0.8GB",
                "quantization": ["4-bit", "8-bit"],
                "finetuned_from": "OLMo-1B",
                "dataset_tags": ["research", "instruction-dataset"]
            }
        ]
        
        # Filter based on query
        if query:
            query_lower = query.lower()
            model_templates = [
                m for m in model_templates
                if query_lower in m["name"].lower()
                or query_lower in m["family"].lower()
                or any(query_lower in tag.lower() for tag in m["tags"])
            ]
        
        # Apply filters
        if filters:
            if "min_params" in filters:
                model_templates = [m for m in model_templates 
                                 if m.get("parameter_count", 0) >= filters["min_params"]]
            
            if "max_params" in filters:
                model_templates = [m for m in model_templates 
                                 if m.get("parameter_count", 0) <= filters["max_params"]]
            
            if "license" in filters:
                model_templates = [m for m in model_templates 
                                 if m.get("license", "").lower() == filters["license"].lower()]
            
            if "tags" in filters:
                model_templates = [m for m in model_templates 
                                 if any(tag in filters["tags"] for tag in m.get("tags", []))]
        
        # Sort by downloads/likes score
        model_templates.sort(key=lambda m: (m.get("downloads", 0) + m.get("likes", 0)) / 1000, reverse=True)
        
        # Return top N models
        return model_templates[:max_models]


class ModelMetadata:
    """Model metadata information."""
    
    def __init__(self, model_data: Dict[str, Any]):
        self.id = model_data.get("id", "")
        self.model_id = model_data.get("model_id", self.id)
        self.name = model_data.get("name", "")
        self.architecture = model_data.get("architecture", "")
        self.family = model_data.get("family", "")
        self.parameter_count = model_data.get("parameter_count", 0)
        self.size = model_data.get("size", 0)
        self.context_length = model_data.get("context_length", 0)
        self.license = model_data.get("license", "")
        self.tags = model_data.get("tags", [])
        self.downloads = model_data.get("downloads", 0)
        self.likes = model_data.get("likes", 0)
        self.pipeline_tag = model_data.get("pipeline_tag", "")
        self.library_name = model_data.get("library_name", "")
        self.model_size = model_data.get("model_size", "")
        self.quantization = model_data.get("quantization", [])
        self.finetuned_from = model_data.get("finetuned_from", "")
        self.dataset_tags = model_data.get("dataset_tags", [])
        self.model_score = model_data.get("model_score", 0.0)
        self.created_at = time.time()
        self.last_updated = time.time()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create ModelMetadata from dictionary."""
        return cls(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelMetadata to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "name": self.name,
            "architecture": self.architecture,
            "family": self.family,
            "parameter_count": self.parameter_count,
            "size": self.size,
            "context_length": self.context_length,
            "license": self.license,
            "tags": self.tags,
            "downloads": self.downloads,
            "likes": self.likes,
            "pipeline_tag": self.pipeline_tag,
            "library_name": self.library_name,
            "model_size": self.model_size,
            "quantization": self.quantization,
            "finetuned_from": self.finetuned_from,
            "dataset_tags": self.dataset_tags,
            "model_score": self.model_score,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }
    
    def get_size_gb(self) -> float:
        """Get model size in GB."""
        return self.size
    
    def get_params_billions(self) -> float:
        """Get model parameters in billions."""
        return self.parameter_count / 1000000000.0
    
    def is_instruction_tuned(self) -> bool:
        """Check if model is instruction tuned."""
        return any("instruction" in tag.lower() for tag in self.tags)
    
    def is_conversational(self) -> bool:
        """Check if model is conversational."""
        return any("conversational" in tag.lower() for tag in self.tags)
    
    def is_code_generation(self) -> bool:
        """Check if model is good for code generation."""
        return any("code" in tag.lower() for tag in self.tags)
    
    def get_relevance_score(self, query: str) -> float:
        """Calculate relevance score for a query."""
        score = 0.0
        query_lower = query.lower()
        
        # Check for exact name match
        if query_lower == self.name.lower():
            score += 10.0
        
        # Check for family match
        if query_lower == self.family.lower():
            score += 8.0
        
        # Check for tag matches
        tag_matches = sum(1 for tag in self.tags if query_lower in tag.lower())
        score += tag_matches * 3.0
        
        # Check for parameter count relevance (if query mentions parameters)
        if "param" in query_lower or "parameter" in query_lower:
            if self.parameter_count >= 1000000000:  # 1B+
                score += 5.0
            elif self.parameter_count >= 100000000:  # 100M
                score += 3.0
            else:
                score += 1.0
        
        # Normalize score
        score = min(score, 10.0)
        
        return score