"""
Task planner for NEXUS autonomous orchestration.

Parses user natural-language requests into structured training objectives,
creates reproducible execution plans, and manages the workflow definition.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus.core.exceptions import ValidationError
from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


@dataclass
class TrainingObjective:
    """Structured training objective parsed from user request."""
    
    objective: str
    max_parameters: Optional[str] = None
    languages: List[str] = field(default_factory=list)
    chat_capability: bool = False
    code_capability: bool = False
    training_method: Optional[str] = None
    preferred_providers: List[str] = field(default_factory=list)
    model_registry: str = "huggingface"
    budget: str = "free"
    
    # Optional fields
    context_length: Optional[int] = None
    license_preference: Optional[str] = None
    quantization: Optional[str] = None
    target_datasets: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "max_parameters": self.max_parameters,
            "languages": self.languages,
            "chat_capability": self.chat_capability,
            "code_capability": self.code_capability,
            "training_method": self.training_method,
            "preferred_providers": self.preferred_providers,
            "model_registry": self.model_registry,
            "budget": self.budget,
            "context_length": self.context_length,
            "license_preference": self.license_preference,
            "quantization": self.quantization,
            "target_datasets": self.target_datasets,
        }


@dataclass
class ExecutionPlan:
    """Reproducible execution plan for model development."""
    
    plan_id: str
    objective: TrainingObjective
    steps: List[Dict[str, Any]]
    estimated_duration_minutes: int
    estimated_gpu_hours: float
    provider_sequence: List[str]
    checkpoint_strategy: str
    created_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "objective": self.objective.to_dict(),
            "steps": self.steps,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "estimated_gpu_hours": self.estimated_gpu_hours,
            "provider_sequence": self.provider_sequence,
            "checkpoint_strategy": self.checkpoint_strategy,
            "created_at": self.created_at,
        }
    
    def get_plan_hash(self) -> str:
        """Get a deterministic hash of the plan for reproducibility.
        
        Excludes plan_id and created_at since those vary between runs.
        """
        # Create hash-relevant dict excluding non-content fields
        hash_dict = {
            "objective": self.objective.to_dict(),
            "steps": self.steps,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "estimated_gpu_hours": self.estimated_gpu_hours,
            "provider_sequence": self.provider_sequence,
            "checkpoint_strategy": self.checkpoint_strategy,
        }
        plan_json = json.dumps(hash_dict, sort_keys=True)
        return hashlib.sha256(plan_json.encode()).hexdigest()[:16]


class WorkflowPlanner:
    """Plans and creates execution workflows for model development."""
    
    def __init__(
        self,
        model_discovery=None,
        dataset_discovery=None,
    ):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.model_discovery = model_discovery
        self.dataset_discovery = dataset_discovery
    
    def parse_intent(self, user_request: str) -> TrainingObjective:
        """Parse a natural-language request into a structured training objective."""
        self.nexus_logger.log_event(
            "intent_parsing_start",
            {
                "user_request": user_request[:200],
            }
        )
        
        request_lower = user_request.lower()
        
        # Extract objective
        objective = user_request.strip()
        
        # Extract max parameters
        max_parameters = None
        import re
        
        # Look for patterns like "under 1B", "1 billion", "500M", "0.5B"
        param_patterns = [
            r'under\s+(\d+(?:\.\d+)?)\s*(b|billion|m|million)',
            r'(\d+(?:\.\d+)?)\s*(b|billion|m|million)',
            r'(\d+(?:\.\d+)?)\s*params',
            r'(\d+(?:\.\d+)?)\s*parameters',
        ]
        
        for pattern in param_patterns:
            match = re.search(pattern, request_lower)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                if unit in ('b', 'billion'):
                    # Format as integer if whole number, otherwise keep decimal
                    if value == int(value):
                        max_parameters = f"{int(value)}B"
                    else:
                        max_parameters = f"{value}B"
                elif unit in ('m', 'million'):
                    val_b = value / 1000
                    if val_b == int(val_b):
                        max_parameters = f"{int(val_b)}B"
                    else:
                        max_parameters = f"{val_b}B"
                break
        
        # Extract languages
        languages = []
        language_keywords = {
            'python': 'python',
            'javascript': 'javascript',
            'js': 'javascript',
            'java': 'java',
            'cpp': 'cpp',
            'c++': 'cpp',
            'go': 'go',
            'rust': 'rust',
            'ruby': 'ruby',
            'php': 'php',
            'swift': 'swift',
            'kotlin': 'kotlin',
        }
        
        for keyword, lang in language_keywords.items():
            if keyword in request_lower and lang not in languages:
                languages.append(lang)
        
        # Extract capabilities
        chat_capability = any(
            word in request_lower for word in ['chat', 'conversational', 'dialogue', 'assistant']
        )
        
        code_capability = any(
            word in request_lower for word in ['code', 'coding', 'programming', 'python', 'javascript']
        )
        
        # Determine training method
        training_method = None
        if 'qlora' in request_lower:
            training_method = 'qlora'
        elif 'lora' in request_lower:
            training_method = 'lora'
        elif 'sft' in request_lower:
            training_method = 'sft'
        elif 'peft' in request_lower:
            training_method = 'peft'
        else:
            # Default based on parameter size
            if max_parameters:
                try:
                    val = float(max_parameters.replace('B', ''))
                    if val <= 1.0:
                        training_method = 'qlora'
                    elif val <= 7.0:
                        training_method = 'qlora'
                    else:
                        training_method = 'qlora'
                except ValueError:
                    training_method = 'qlora'
            else:
                training_method = 'qlora'
        
        # Determine preferred providers
        preferred_providers = ['kaggle', 'colab']
        if 'huggingface' in request_lower or 'hf' in request_lower:
            preferred_providers.insert(0, 'huggingface')
        
        # Extract context length
        context_length = None
        ctx_match = re.search(r'context.*?(\d+)', request_lower)
        if ctx_match:
            context_length = int(ctx_match.group(1))
        
        # Extract license preference
        license_preference = None
        for lic in ['apache', 'mit', 'bsd', 'cc-by', 'llama2']:
            if lic in request_lower:
                license_preference = lic.upper() if lic != 'llama2' else 'LLAMA2'
                break
        
        # Extract quantization
        quantization = None
        for q in ['4-bit', '4bit', '8-bit', '8bit', 'nf4', 'int8']:
            if q in request_lower:
                quantization = q
                break
        
        # Build objective
        objective_obj = TrainingObjective(
            objective=objective,
            max_parameters=max_parameters,
            languages=languages,
            chat_capability=chat_capability,
            code_capability=code_capability,
            training_method=training_method,
            preferred_providers=preferred_providers,
            model_registry='huggingface',
            budget='free',
            context_length=context_length,
            license_preference=license_preference,
            quantization=quantization,
        )
        
        self.nexus_logger.log_event(
            "intent_parsing_completed",
            {
                "objective": objective_obj.to_dict(),
            }
        )
        
        return objective_obj
    
    def plan(
        self,
        objective: str,
        model_requirements: Dict[str, Any],
        dataset_requirements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Plan a complete execution workflow."""
        self.nexus_logger.log_event(
            "workflow_planning_start",
            {
                "objective": objective,
                "model_requirements": model_requirements,
                "dataset_requirements": dataset_requirements,
            }
        )
        
        # Create TrainingObjective from requirements
        training_objective = TrainingObjective(
            objective=objective,
            max_parameters=model_requirements.get("max_parameters"),
            languages=model_requirements.get("languages", []),
            chat_capability=model_requirements.get("capabilities", {}).get("conversational", False),
            code_capability=model_requirements.get("capabilities", {}).get("code_generation", False),
            training_method=model_requirements.get("training_method", "qlora"),
            preferred_providers=model_requirements.get("preferred_providers", ["kaggle", "colab"]),
            model_registry=model_requirements.get("model_registry", "huggingface"),
            budget=model_requirements.get("budget", "free"),
        )
        
        # Build execution steps
        steps = self._build_execution_steps(training_objective)
        
        # Estimate duration and resources
        estimated_duration = self._estimate_duration(training_objective, steps)
        estimated_gpu_hours = self._estimate_gpu_hours(training_objective)
        
        # Determine provider sequence
        provider_sequence = training_objective.preferred_providers.copy()
        
        # Create plan
        plan_id = f"plan_{int(time.time())}_{hashlib.md5(objective.encode()).hexdigest()[:8]}"
        
        plan = ExecutionPlan(
            plan_id=plan_id,
            objective=training_objective,
            steps=steps,
            estimated_duration_minutes=estimated_duration,
            estimated_gpu_hours=estimated_gpu_hours,
            provider_sequence=provider_sequence,
            checkpoint_strategy="epoch_level_hf_sync",
            created_at=time.time(),
        )
        
        self.nexus_logger.log_event(
            "workflow_planning_completed",
            {
                "plan_id": plan_id,
                "steps_count": len(steps),
                "estimated_duration_minutes": estimated_duration,
                "estimated_gpu_hours": estimated_gpu_hours,
                "plan_hash": plan.get_plan_hash(),
            }
        )
        
        return plan.to_dict()
    
    def _build_execution_steps(
        self,
        objective: TrainingObjective,
    ) -> List[Dict[str, Any]]:
        """Build the ordered list of execution steps."""
        steps = []
        
        step_order = [
            ("intent_parsed", "Intent parsed into structured objective", "completed"),
            ("model_discovery", "Discover suitable base models from Hugging Face", "pending"),
            ("model_selection", "Select best model based on requirements", "pending"),
            ("dataset_discovery", "Discover suitable datasets", "pending"),
            ("dataset_validation", "Validate dataset licenses and usability", "pending"),
            ("data_preparation", "Clean, format, and split training data", "pending"),
            ("preflight_check", "Verify GPU feasibility and resource requirements", "pending"),
            ("provider_selection", "Select optimal cloud provider", "pending"),
            ("training_job_submit", "Submit training job to cloud provider", "pending"),
            ("training_monitoring", "Monitor training execution and checkpoints", "pending"),
            ("model_evaluation", "Evaluate baseline and fine-tuned model", "pending"),
            ("improvement_loop", "Additional training if justified", "pending"),
            ("model_export", "Export trained model and adapter", "pending"),
            ("report_generation", "Generate complete training report", "pending"),
            ("publish_optional", "Publish to Hugging Face (requires approval)", "pending"),
        ]
        
        for step_id, description, status in step_order:
            steps.append({
                "step_id": step_id,
                "description": description,
                "status": status,
                "depends_on": steps[-1]["step_id"] if steps else None,
                "estimated_duration_minutes": self._estimate_step_duration(step_id),
            })
        
        return steps
    
    def _estimate_step_duration(self, step_id: str) -> int:
        """Estimate duration for a single step in minutes."""
        durations = {
            "intent_parsed": 0,
            "model_discovery": 2,
            "model_selection": 1,
            "dataset_discovery": 3,
            "dataset_validation": 2,
            "data_preparation": 10,
            "preflight_check": 1,
            "provider_selection": 1,
            "training_job_submit": 5,
            "training_monitoring": 60,
            "model_evaluation": 15,
            "improvement_loop": 30,
            "model_export": 5,
            "report_generation": 5,
            "publish_optional": 0,
        }
        return durations.get(step_id, 10)
    
    def _estimate_duration(
        self,
        objective: TrainingObjective,
        steps: List[Dict[str, Any]],
    ) -> int:
        """Estimate total duration in minutes."""
        return sum(step.get("estimated_duration_minutes", 0) for step in steps)
    
    def _estimate_gpu_hours(
        self,
        objective: TrainingObjective,
    ) -> float:
        """Estimate GPU hours required."""
        # Estimate based on parameter size and method
        param_val = 0.5  # Default 0.5B
        if objective.max_parameters:
            try:
                param_val = float(objective.max_parameters.replace('B', ''))
            except ValueError:
                pass
        
        method = objective.training_method or "qlora"
        
        # Base hours per parameter billion
        hours_per_billion = {
            "qlora": 2.0,
            "lora": 4.0,
            "sft": 8.0,
            "peft": 3.0,
        }
        
        base_hours = hours_per_billion.get(method, 2.0) * param_val
        
        # Add overhead for data prep and evaluation
        total_hours = base_hours + 2.0
        
        return round(total_hours, 1)
    
    def create_reproducible_plan(
        self,
        user_request: str,
    ) -> Dict[str, Any]:
        """Create a complete reproducible plan from a user request."""
        # Parse intent
        objective = self.parse_intent(user_request)
        
        # Build requirements
        capabilities = {}
        if objective.chat_capability:
            capabilities["conversational"] = True
        if objective.code_capability:
            capabilities["code_generation"] = True
        
        model_requirements = {
            "objective": user_request,
            "max_parameters": objective.max_parameters,
            "languages": objective.languages,
            "capabilities": capabilities,
        }
        
        model_requirements["training_method"] = objective.training_method
        model_requirements["preferred_providers"] = objective.preferred_providers
        model_requirements["model_registry"] = objective.model_registry
        model_requirements["budget"] = objective.budget
        
        dataset_requirements = {
            "objective": user_request,
            "types": ["instruction", "code", "conversational"],
            "languages": objective.languages,
        }
        
        # Create plan
        plan = self.plan(
            objective=user_request,
            model_requirements=model_requirements,
            dataset_requirements=dataset_requirements,
        )
        
        return plan