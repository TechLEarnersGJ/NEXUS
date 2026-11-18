"""
Model metadata for NEXUS.

Manages model metadata, including information about model architecture,
parameters, capabilities, and training details.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from nexus.core.exceptions import ValidationError


@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    
    # Model identification
    model_id: str
    model_name: str
    model_path: Optional[str] = None
    family: Optional[str] = None
    version: Optional[str] = None
    
    # Architecture
    architecture: Optional[str] = None
    framework: Optional[str] = None
    model_type: Optional[str] = None
    
    # Parameters
    parameter_count: Optional[int] = None
    size_bytes: Optional[int] = None
    context_length: Optional[int] = None
    vocab_size: Optional[int] = None
    
    # Licensing and usage
    license: Optional[str] = None
    license_url: Optional[str] = None
    requires_authentication: bool = True
    
    # Capabilities
    languages: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    
    # Quality indicators
    downloads: Optional[int] = None
    likes: Optional[int] = None
    model_score: Optional[float] = None
    community_rating: Optional[float] = None
    
    # Training information
    trained_on_datasets: List[str] = field(default_factory=list)
    training_frameworks: List[str] = field(default_factory=list)
    training_parameters: Optional[Dict[str, Any]] = None
    finetuned_from: Optional[str] = None
    
    # Model information
    created_at: Optional[float] = None
    last_updated: Optional[float] = None
    author: Optional[str] = None
    paper_url: Optional[str] = None
    demo_url: Optional[str] = None
    
    # File information
    files: List[str] = field(default_factory=list)
    model_files: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    tokenizer_files: List[str] = field(default_factory=list)
    
    # Storage
    storage_path: Optional[str] = None
    is_cached: bool = False
    cache_date: Optional[float] = None
    
    # Metadata
    source: Optional[str] = None
    source_url: Optional[str] = None
    verification_status: Optional[str] = None
    
    def __post_init__(self):
        """Validate model metadata after initialization."""
        self._validate_metadata()
    
    def _validate_metadata(self):
        """Validate model metadata."""
        errors = []
        
        if not self.model_id:
            errors.append("Model ID cannot be empty")
        
        if not self.model_name:
            errors.append("Model name cannot be empty")
        
        if self.parameter_count is not None and self.parameter_count <= 0:
            errors.append("Parameter count must be positive")
        
        if self.size_bytes is not None and self.size_bytes <= 0:
            errors.append("Size must be positive")
        
        if self.context_length is not None and self.context_length <= 0:
            errors.append("Context length must be positive")
        
        if self.license and not self.license.strip():
            errors.append("License cannot be empty if specified")
        
        if errors:
            raise ValidationError("; ".join(errors))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create ModelMetadata from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_huggingface_api(cls, api_data: Dict[str, Any]) -> "ModelMetadata":
        """Create ModelMetadata from Hugging Face API response."""
        # Map Hugging Face API fields to our metadata
        metadata = {
            "model_id": api_data.get("id", ""),
            "model_name": api_data.get("modelId", api_data.get("id", "")),
            "family": api_data.get("modelId", "").split("/")[0] if "/" in api_data.get("modelId", "") else "",
            "architecture": api_data.get("modelId", "").split("/")[1] if "/" in api_data.get("modelId", "") else "",
            "parameter_count": cls._parse_parameter_count(api_data),
            "size_bytes": cls._estimate_model_size(api_data),
            "context_length": api_data.get("config", {}).get("max_position_embeddings", 0),
            "vocab_size": api_data.get("config", {}).get("vocab_size", 0),
            "license": api_data.get("license", ""),
            "tags": api_data.get("tags", []),
            "downloads": api_data.get("downloads", 0),
            "likes": api_data.get("likes", 0),
            "model_score": api_data.get("score", 0.0),
            "created_at": time.time(),
            "last_updated": api_data.get("lastModified", time.time()),
            "author": api_data.get("author", ""),
            "paper_url": api_data.get("paperUrl", ""),
            "source": "huggingface",
            "source_url": f"https://huggingface.co/{api_data.get('id', '')}",
            "verification_status": api_data.get("verificationStatus", "unverified")
        }
        
        return cls(**metadata)
    
    @staticmethod
    def _parse_parameter_count(api_data: Dict[str, Any]) -> Optional[int]:
        """Parse parameter count from Hugging Face API data."""
        # Try to get parameter count from model ID
        model_id = api_data.get("modelId", api_data.get("id", ""))
        
        if "0.5B" in model_id:
            return 500000000
        elif "7B" in model_id:
            return 7000000000
        elif "13B" in model_id:
            return 13000000000
        elif "70B" in model_id:
            return 70000000000
        elif "100B" in model_id:
            return 100000000000
        elif "180B" in model_id:
            return 180000000000
        
        # Try to get from config
        config = api_data.get("config", {})
        if "num_parameters" in config:
            return config["num_parameters"]
        
        # Try to get from pretrained_config
        pretrained_config = api_data.get("pretrained_config", {})
        if "num_parameters" in pretrained_config:
            return pretrained_config["num_parameters"]
        
        return None
    
    @staticmethod
    def _estimate_model_size(api_data: Dict[str, Any]) -> Optional[int]:
        """Estimate model size from Hugging Face API data."""
        # Use estimated size based on parameter count
        param_count = ModelMetadata._parse_parameter_count(api_data)
        
        if param_count:
            # Estimate: 8 bytes per parameter for float32
            # With quantization and compression, reduce by factor of 4
            estimated_bytes = param_count * 8 / 4
            
            # Add overhead for tokenizer and metadata
            estimated_bytes += 50 * 1024 * 1024  # 50MB overhead
            
            return int(estimated_bytes)
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelMetadata to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "family": self.family,
            "version": self.version,
            "architecture": self.architecture,
            "framework": self.framework,
            "model_type": self.model_type,
            "parameter_count": self.parameter_count,
            "size_bytes": self.size_bytes,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "license": self.license,
            "license_url": self.license_url,
            "requires_authentication": self.requires_authentication,
            "languages": self.languages,
            "capabilities": self.capabilities,
            "tags": self.tags,
            "use_cases": self.use_cases,
            "downloads": self.downloads,
            "likes": self.likes,
            "model_score": self.model_score,
            "community_rating": self.community_rating,
            "trained_on_datasets": self.trained_on_datasets,
            "training_frameworks": self.training_frameworks,
            "training_parameters": self.training_parameters,
            "finetuned_from": self.finetuned_from,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "author": self.author,
            "paper_url": self.paper_url,
            "demo_url": self.demo_url,
            "files": self.files,
            "model_files": self.model_files,
            "config_files": self.config_files,
            "tokenizer_files": self.tokenizer_files,
            "storage_path": self.storage_path,
            "is_cached": self.is_cached,
            "cache_date": self.cache_date,
            "source": self.source,
            "source_url": self.source_url,
            "verification_status": self.verification_status
        }
    
    def get_size_gb(self) -> Optional[float]:
        """Get model size in GB."""
        if self.size_bytes:
            return self.size_bytes / (1024 ** 3)
        return None
    
    def get_params_billions(self) -> Optional[float]:
        """Get model parameters in billions."""
        if self.parameter_count:
            return self.parameter_count / (1000 ** 9)
        return None
    
    def is_instruction_tuned(self) -> bool:
        """Check if model is instruction tuned."""
        return any("instruction" in tag.lower() for tag in self.tags)
    
    def is_conversational(self) -> bool:
        """Check if model is conversational."""
        return any("conversational" in tag.lower() for tag in self.tags)
    
    def is_code_generation(self) -> bool:
        """Check if model is good for code generation."""
        return any("code" in tag.lower() for tag in self.tags)
    
    def is_multilingual(self) -> bool:
        """Check if model supports multiple languages."""
        return len(self.languages) > 1
    
    def is_large_model(self, threshold_b: float = 1.0) -> bool:
        """Check if model is large (more than threshold billion parameters)."""
        if self.parameter_count:
            return self.parameter_count > threshold_b * 1000000000
        return False
    
    def needs_gpu(self) -> bool:
        """Check if model requires GPU for inference."""
        if self.parameter_count:
            return self.parameter_count > 100000000  # 100M parameters
        return False
    
    def get_relevance_score(self, query: str) -> float:
        """Calculate relevance score for a query."""
        score = 0.0
        query_lower = query.lower()
        
        # Check for exact name match
        if query_lower == self.model_name.lower():
            score += 10.0
        
        # Check for family match
        if self.family and query_lower == self.family.lower():
            score += 8.0
        
        # Check for tag matches
        tag_matches = sum(1 for tag in self.tags if query_lower in tag.lower())
        score += tag_matches * 3.0
        
        # Check for parameter count relevance (if query mentions parameters)
        if "param" in query_lower or "parameter" in query_lower:
            if self.parameter_count and self.parameter_count >= 1000000000:  # 1B+
                score += 5.0
            elif self.parameter_count and self.parameter_count >= 100000000:  # 100M
                score += 3.0
            else:
                score += 1.0
        
        # Normalize score
        score = min(score, 10.0)
        
        return score


class ModelRegistry:
    """Registry for model metadata."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "workspace/models_registry")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._cache: Dict[str, ModelMetadata] = {}
        
        # Registry file
        self._registry_file = self.storage_path / "model_registry.json"
        
        # Load registry
        self._load_registry()
    
    def _load_registry(self):
        """Load model registry from disk."""
        if self._registry_file.exists():
            try:
                with open(self._registry_file, 'r') as f:
                    import json
                    data = json.load(f)
                
                # Convert to ModelMetadata objects
                for model_id, model_data in data.items():
                    try:
                        self._cache[model_id] = ModelMetadata.from_dict(model_data)
                    except Exception as e:
                        print(f"Warning: Failed to load model {model_id}: {str(e)}")
                        
            except Exception as e:
                print(f"Warning: Failed to load model registry: {str(e)}")
    
    def save_model(self, model: ModelMetadata):
        """Save model metadata to registry."""
        self._cache[model.model_id] = model
        
        # Save to disk
        try:
            with open(self._registry_file, 'w') as f:
                import json
                data = {model_id: model.to_dict() for model_id, model in self._cache.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save model registry: {str(e)}")
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model by ID."""
        return self._cache.get(model_id)
    
    def remove_model(self, model_id: str):
        """Remove model from registry."""
        if model_id in self._cache:
            del self._cache[model_id]
        
        # Save to disk
        try:
            with open(self._registry_file, 'w') as f:
                import json
                data = {model_id: model.to_dict() for model_id, model in self._cache.items()}
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def list_models(self, family: Optional[str] = None, 
                    license: Optional[str] = None) -> List[ModelMetadata]:
        """List models with optional filtering."""
        models = list(self._cache.values())
        
        if family:
            models = [m for m in models if m.family and m.family.lower() == family.lower()]
        
        if license:
            models = [m for m in models if m.license and m.license.lower() == license.lower()]
        
        return models
    
    def search_models(self, query: str, max_results: int = 10) -> List[ModelMetadata]:
        """Search models by query."""
        models = self.list_models()
        
        # Calculate relevance scores
        scored_models = []
        for model in models:
            score = model.get_relevance_score(query)
            scored_models.append((model, score))
        
        # Sort by score (descending)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        return [model for model, _ in scored_models[:max_results]]