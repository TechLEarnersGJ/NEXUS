"""
Model selector for NEXUS.

Selects the best model for a given use case based on various criteria
like parameters, license, capabilities, and user requirements.
"""

from typing import List, Optional, Dict, Any
from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger
from nexus.models.discovery import ModelMetadata


class ModelSelector:
    """Selects the best model for a given use case."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
    
    def select_best_model(self, models: List[ModelMetadata], 
                         requirements: Dict[str, Any]) -> Optional[ModelMetadata]:
        """Select the best model based on requirements."""
        self.nexus_logger.log_event(
            "model_selection",
            {
                "total_models": len(models),
                "requirements": requirements
            }
        )
        
        if not models:
            return None
        
        # Filter models based on requirements
        filtered_models = self._filter_models(models, requirements)
        
        if not filtered_models:
            # If no models match requirements, return the highest quality one
            self.logger.warning("No models match all requirements, returning best available")
            return max(filtered_models, key=lambda m: m.get_relevance_score(
                requirements.get("objective", "")
            )) if filtered_models else None
        
        # Score models based on requirements
        scored_models = []
        for model in filtered_models:
            score = self._score_model(model, requirements)
            scored_models.append((model, score))
        
        # Select the best model
        if scored_models:
            best_model, best_score = max(scored_models, key=lambda x: x[1])
            self.logger.info(f"Selected model: {best_model.name} (score: {best_score:.2f})")
            return best_model
        
        return None
    
    def _filter_models(self, models: List[ModelMetadata], 
                       requirements: Dict[str, Any]) -> List[ModelMetadata]:
        """Filter models based on requirements."""
        filtered_models = []
        
        for model in models:
            # Check parameter count requirements
            if "max_parameters" in requirements:
                if model.parameter_count > requirements["max_parameters"]:
                    continue
            
            if "min_parameters" in requirements:
                if model.parameter_count < requirements["min_parameters"]:
                    continue
            
            # Check license requirements
            if "required_license" in requirements:
                if model.license.lower() != requirements["required_license"].lower():
                    continue
            
            # Check language requirements
            if "languages" in requirements:
                # Note: This is a simplified check; actual language support
                # would be determined from model training data
                pass
            
            # Check capability requirements
            if "capabilities" in requirements:
                if not self._model_has_capabilities(model, requirements["capabilities"]):
                    continue
            
            filtered_models.append(model)
        
        return filtered_models
    
    def _model_has_capabilities(self, model: ModelMetadata, 
                                required_capabilities: List[str]) -> bool:
        """Check if model has required capabilities."""
        # This is a simplified implementation
        # In practice, you would have a more sophisticated capability matching system
        
        for capability in required_capabilities:
            capability_lower = capability.lower()
            
            if capability_lower == "conversational":
                if not model.is_conversational():
                    return False
            
            elif capability_lower == "instruction_following":
                if not model.is_instruction_tuned():
                    return False
            
            elif capability_lower == "code_generation":
                if not model.is_code_generation():
                    return False
            
            elif capability_lower == "multilingual":
                # Check if model supports multiple languages
                if "multilingual" not in [tag.lower() for tag in model.tags]:
                    return False
        
        return True
    
    def _score_model(self, model: ModelMetadata, 
                     requirements: Dict[str, Any]) -> float:
        """Score a model based on requirements."""
        score = 0.0
        
        # Parameter count score
        if "target_parameters" in requirements:
            target_params = requirements["target_parameters"]
            param_diff = abs(model.parameter_count - target_params)
            param_score = max(0, 10 - (param_diff / target_params))
            score += param_score * 0.3
        
        # License compatibility
        if "license" in requirements and "license_weight" in requirements:
            if model.license.lower() == requirements["license"].lower():
                score += requirements["license_weight"] * 2.0
        
        # Capability alignment
        if "capabilities" in requirements:
            capability_score = self._calculate_capability_score(model, requirements["capabilities"])
            score += capability_score * 0.4
        
        # Size/bandwidth considerations
        if "max_size_gb" in requirements:
            if model.size <= requirements["max_size_gb"]:
                score += 5.0
            else:
                score -= 2.0
        
        # Quality score (based on downloads, likes, model score)
        quality_score = (model.downloads / 10000) + (model.likes / 100) + model.model_score
        score += min(quality_score, 10.0) * 0.2
        
        # Query relevance
        if "objective" in requirements:
            relevance = model.get_relevance_score(requirements["objective"])
            score += relevance * 0.1
        
        return min(score, 100.0)  # Cap at 100
    
    def _calculate_capability_score(self, model: ModelMetadata, 
                                    required_capabilities: List[str]) -> float:
        """Calculate capability alignment score."""
        if not required_capabilities:
            return 10.0
        
        matched_capabilities = 0
        for capability in required_capabilities:
            if self._model_has_capabilities(model, [capability]):
                matched_capabilities += 1
        
        return (matched_capabilities / len(required_capabilities)) * 10.0