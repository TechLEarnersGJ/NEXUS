"""
Reasoning evaluation benchmarks for NEXUS.

Evaluates basic reasoning capabilities including:
- Logical reasoning
- Math word problems
- Common sense reasoning
"""

from typing import Any, Dict, List, Optional

from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


class ReasoningEvaluator:
    """Evaluates reasoning capabilities of models."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        
        # Reasoning test cases
        self.reasoning_cases = [
            {
                "name": "syllogism",
                "prompt": "All humans are mortal. Socrates is human. Is Socrates mortal?",
                "expected_answer": "yes",
                "category": "logical",
            },
            {
                "name": "math_word_problem",
                "prompt": "If I have 5 apples and give away 2, how many do I have left?",
                "expected_answer": "3",
                "category": "math",
            },
            {
                "name": "common_sense",
                "prompt": "Can you use a spoon to cut a steak?",
                "expected_answer": "no",
                "category": "commonsense",
            },
            {
                "name": "analogy",
                "prompt": "Dog is to puppy as cat is to what?",
                "expected_answer": "kitten",
                "category": "analogy",
            },
            {
                "name": "negation",
                "prompt": "Which is not a fruit: apple, banana, carrot, orange?",
                "expected_answer": "carrot",
                "category": "classification",
            },
        ]
    
    def evaluate_reasoning(
        self,
        model_id: str,
        response: str,
        test_case_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a reasoning response."""
        self.nexus_logger.log_event(
            "reasoning_eval_start",
            {
                "model_id": model_id,
                "test_case": test_case_name,
            }
        )
        
        results = []
        
        for tc in self.reasoning_cases:
            if test_case_name and tc["name"] != test_case_name:
                continue
            
            # Check if expected answer is in response
            response_lower = response.lower()
            expected = tc["expected_answer"].lower()
            answer_found = expected in response_lower
            
            # Check for contradictory answers
            contradiction_indicators = ["no", "not", "incorrect", "wrong"]
            has_contradiction = any(
                ind in response_lower for ind in contradiction_indicators
            ) and answer_found
            
            result = {
                "test_name": tc["name"],
                "prompt": tc["prompt"],
                "category": tc["category"],
                "expected_answer": tc["expected_answer"],
                "answer_found": answer_found,
                "has_contradiction": has_contradiction,
                "correct": answer_found and not has_contradiction,
            }
            
            results.append(result)
        
        # Calculate overall metrics
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        
        overall = {
            "model_id": model_id,
            "total_tests": total,
            "correct": correct,
            "accuracy": correct / max(total, 1),
            "category_breakdown": self._category_breakdown(results),
            "test_results": results,
        }
        
        self.nexus_logger.log_event(
            "reasoning_eval_completed",
            {
                "model_id": model_id,
                "accuracy": overall["accuracy"],
            }
        )
        
        return overall
    
    def _category_breakdown(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Break down results by category."""
        breakdown = {}
        
        for r in results:
            cat = r["category"]
            if cat not in breakdown:
                breakdown[cat] = {"correct": 0, "total": 0}
            breakdown[cat]["total"] += 1
            if r["correct"]:
                breakdown[cat]["correct"] += 1
        
        for cat in breakdown:
            breakdown[cat]["accuracy"] = round(
                breakdown[cat]["correct"] / max(breakdown[cat]["total"], 1), 3
            )
        
        return breakdown
    
    def evaluate_math_word_problem(
        self,
        model_id: str,
        response: str,
        problem: str,
        expected_answer: str,
    ) -> Dict[str, Any]:
        """Evaluate a math word problem response."""
        response_lower = response.lower()
        expected_lower = expected_answer.lower()
        
        answer_found = expected_lower in response_lower
        
        # Extract numbers from response
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        
        result = {
            "model_id": model_id,
            "problem": problem,
            "expected_answer": expected_answer,
            "answer_found": answer_found,
            "numbers_extracted": numbers,
            "correct": answer_found,
        }
        
        return result