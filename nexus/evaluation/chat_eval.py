"""
Chat evaluation benchmarks for NEXUS.

Evaluates conversational quality of models including:
- Instruction following
- Conversational coherence
- Helpfulness and relevance
"""

from typing import Any, Dict, List, Optional

from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


class ChatEvaluator:
    """Evaluates conversational quality of models."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        
        # Chat test scenarios
        self.chat_scenarios = [
            {
                "name": "greeting",
                "prompt": "Hello! How are you today?",
                "expected_elements": ["hello", "hi", "good", "well", "fine"],
                "category": "social",
            },
            {
                "name": "explain_concept",
                "prompt": "Explain what a variable is in programming, simply.",
                "expected_elements": ["variable", "store", "value", "name", "memory"],
                "category": "educational",
            },
            {
                "name": "code_help",
                "prompt": "Help me debug this Python code: print('hello')  # missing closing quote",
                "expected_elements": ["quote", "syntax", "error", "fix", "missing"],
                "category": "coding",
            },
            {
                "name": "creative_writing",
                "prompt": "Write a short two-sentence story about a robot.",
                "expected_elements": ["robot", "sentence"],
                "category": "creative",
            },
            {
                "name": "multi_step",
                "prompt": "First, tell me the capital of France. Then, tell me 2+2.",
                "expected_elements": ["paris", "4"],
                "category": "reasoning",
            },
        ]
    
    def evaluate_chat_response(
        self,
        model_id: str,
        prompt: str,
        response: str,
        scenario_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single chat response."""
        self.nexus_logger.log_event(
            "chat_eval_start",
            {
                "model_id": model_id,
                "prompt": prompt[:100],
                "scenario": scenario_name,
            }
        )
        
        # Find matching scenario
        scenario = None
        if scenario_name:
            for s in self.chat_scenarios:
                if s["name"] == scenario_name:
                    scenario = s
                    break
        
        # Evaluate response
        eval_result = self._analyze_response(response, scenario)
        
        result = {
            "model_id": model_id,
            "prompt": prompt,
            "response": response,
            "scenario": scenario_name,
            "category": scenario["category"] if scenario else "unknown",
            "score": eval_result["score"],
            "checks": eval_result["checks"],
            "feedback": eval_result["feedback"],
        }
        
        self.nexus_logger.log_event(
            "chat_eval_completed",
            {
                "model_id": model_id,
                "score": result["score"],
            }
        )
        
        return result
    
    def evaluate_conversation(
        self,
        model_id: str,
        conversation: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Evaluate a multi-turn conversation."""
        self.nexus_logger.log_event(
            "conversation_eval_start",
            {
                "model_id": model_id,
                "turns": len(conversation),
            }
        )
        
        turn_results = []
        for i, turn in enumerate(conversation):
            result = self.evaluate_chat_response(
                model_id=model_id,
                prompt=turn.get("prompt", ""),
                response=turn.get("response", ""),
            )
            turn_results.append(result)
        
        # Calculate overall metrics
        avg_score = sum(r["score"] for r in turn_results) / max(len(turn_results), 1)
        
        overall = {
            "model_id": model_id,
            "total_turns": len(turn_results),
            "average_score": avg_score,
            "turn_results": turn_results,
            "category_scores": self._calculate_category_scores(turn_results),
        }
        
        self.nexus_logger.log_event(
            "conversation_eval_completed",
            {
                "model_id": model_id,
                "average_score": avg_score,
            }
        )
        
        return overall
    
    def _analyze_response(
        self,
        response: str,
        scenario: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze a chat response for quality."""
        checks = {}
        feedback = []
        
        # Basic checks
        checks["non_empty"] = len(response.strip()) > 0
        checks["reasonable_length"] = 10 <= len(response.strip()) <= 2000
        checks["no_repetition"] = len(set(response.split())) / max(len(response.split()), 1) > 0.5
        
        if not checks["non_empty"]:
            feedback.append("Response is empty")
        if not checks["reasonable_length"]:
            feedback.append("Response length is unreasonable")
        if not checks["no_repetition"]:
            feedback.append("Response has excessive repetition")
        
        # Scenario-specific checks
        if scenario:
            expected = scenario.get("expected_elements", [])
            response_lower = response.lower()
            
            for element in expected:
                key = f"mentions_{element}"
                checks[key] = element.lower() in response_lower
                if not checks[key]:
                    feedback.append(f"Missing expected element: {element}")
            
            # Check for refusal patterns
            refusal_indicators = [
                "i cannot", "i can't", "i'm not able", "i don't know",
                "sorry, i", "as an ai", "i apologize",
            ]
            checks["no_refusal"] = not any(
                ind in response_lower for ind in refusal_indicators
            )
            if not checks["no_refusal"]:
                feedback.append("Response contains refusal language")
        
        # Calculate score (0.0 to 1.0)
        if checks:
            score = sum(1 for v in checks.values() if v) / len(checks)
        else:
            score = 0.0
        
        return {
            "score": round(score, 3),
            "checks": checks,
            "feedback": feedback,
        }
    
    def _calculate_category_scores(
        self,
        turn_results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate average scores per category."""
        category_scores = {}
        category_counts = {}
        
        for result in turn_results:
            cat = result.get("category", "unknown")
            if cat not in category_scores:
                category_scores[cat] = 0.0
                category_counts[cat] = 0
            category_scores[cat] += result["score"]
            category_counts[cat] += 1
        
        return {
            cat: round(category_scores[cat] / category_counts[cat], 3)
            for cat in category_scores
        }