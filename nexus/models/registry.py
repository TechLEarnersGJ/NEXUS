"""
Model registry for NEXUS.

Manages the registry of available models, including local and remote models,
with capabilities for searching, filtering, and managing model metadata.
"""

import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from nexus.core.logger import get_logger
from nexus.models.discovery import ModelMetadata


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
        models = list(self._cache.values())
        
        # Calculate relevance scores
        scored_models = []
        for model in models:
            score = model.get_relevance_score(query)
            scored_models.append((model, score))
        
        # Sort by score (descending)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        return [model for model, _ in scored_models[:max_results]]


# Alias for backward compatibility
class ModelRegistryAlias(ModelRegistry):
    """Alias for ModelRegistry (deprecated)."""
    
    pass


# Export commonly used classes
__all__ = [
    "ModelMetadata",
    "ModelRegistry",
    "ModelRegistryAlias",
]