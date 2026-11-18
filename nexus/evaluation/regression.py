"""
Regression testing for NEXUS.

Compares baseline and fine-tuned model performance to detect
regressions and improvements across evaluation metrics.
"""

from typing import Any, Dict, List, Optional

from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


class RegressionTester:
    """Compares model versions for regressions and improvements."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
    
    def compare_models(
        self,
        baseline_results: Dict[str, Any],
        fine_tuned_results: Dict[str, Any],
        model_a_name: str = "baseline",
        model_b_name: str = "fine_tuned",
    ) -> Dict[str, Any]:
        """Compare evaluation results between two models."""
        self.nexus_logger.log_event(
            "regression_comparison_start",
            {
                "model_a": model_a_name,
                "model_b": model_b_name,
            }
        )
        
        comparison = {
            "model_a": model_a_name,
            "model_b": model_b_name,
            "metrics_compared": [],
            "regressions": [],
            "improvements": [],
            "unchanged": [],
            "overall_assessment": "unchanged",
        }
        
        # Compare common metrics
        metrics_a = self._extract_metrics(baseline_results)
        metrics_b = self._extract_metrics(fine_tuned_results)
        
        all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())
        
        for metric in all_metrics:
            val_a = metrics_a.get(metric)
            val_b = metrics_b.get(metric)
            
            if val_a is None or val_b is None:
                continue
            
            comparison["metrics_compared"].append(metric)
            
            change = val_b - val_a
            change_percent = (change / max(abs(val_a), 0.001)) * 100
            
            metric_comparison = {
                "metric": metric,
                "model_a_value": val_a,
                "model_b_value": val_b,
                "change": round(change, 4),
                "change_percent": round(change_percent, 2),
            }
            
            # Determine if regression or improvement
            # Higher is better for most metrics (accuracy, pass_rate, etc.)
            # Lower is better for some (loss, error_rate, etc.)
            higher_is_better = metric not in ("loss", "error_rate", "perplexity")
            
            if higher_is_better:
                if change > 0.01:  # >1% improvement
                    comparison["improvements"].append(metric_comparison)
                elif change < -0.01:  # >1% regression
                    comparison["regressions"].append(metric_comparison)
                else:
                    comparison["unchanged"].append(metric_comparison)
            else:
                if change < -0.01:  # Lower loss is better
                    comparison["improvements"].append(metric_comparison)
                elif change > 0.01:  # Higher loss is regression
                    comparison["regressions"].append(metric_comparison)
                else:
                    comparison["unchanged"].append(metric_comparison)
        
        # Overall assessment
        if len(comparison["regressions"]) > len(comparison["improvements"]):
            comparison["overall_assessment"] = "regression"
        elif len(comparison["improvements"]) > len(comparison["regressions"]):
            comparison["overall_assessment"] = "improvement"
        else:
            comparison["overall_assessment"] = "unchanged"
        
        self.nexus_logger.log_event(
            "regression_comparison_completed",
            {
                "assessment": comparison["overall_assessment"],
                "regressions": len(comparison["regressions"]),
                "improvements": len(comparison["improvements"]),
            }
        )
        
        return comparison
    
    def _extract_metrics(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, float]:
        """Extract numeric metrics from evaluation results."""
        metrics = {}
        
        # Common metric paths
        metric_paths = [
            ("pass_rate", ["pass_rate"]),
            ("accuracy", ["accuracy"]),
            ("syntax_valid_rate", ["syntax_valid_rate"]),
            ("safety_score", ["safety_score"]),
            ("overall_score", ["overall_score"]),
            ("average_score", ["average_score"]),
            ("final_loss", ["final_loss"]),
            ("best_loss", ["best_loss"]),
        ]
        
        for metric_name, path in metric_paths:
            value = self._get_nested(results, path)
            if value is not None:
                try:
                    metrics[metric_name] = float(value)
                except (ValueError, TypeError):
                    pass
        
        # Extract from nested test results
        if "test_results" in results:
            for test_result in results["test_results"]:
                if isinstance(test_result, dict):
                    test_name = test_result.get("test_name", "unknown")
                    if "pass_rate" in test_result:
                        metrics[f"test_{test_name}_pass_rate"] = float(test_result["pass_rate"])
                    if "accuracy" in test_result:
                        metrics[f"test_{test_name}_accuracy"] = float(test_result["accuracy"])
        
        return metrics
    
    def _get_nested(
        self,
        data: Dict[str, Any],
        path: List[str],
    ) -> Any:
        """Get a value from a nested dict using a path."""
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def evaluate_improvement(
        self,
        baseline_results: Dict[str, Any],
        fine_tuned_results: Dict[str, Any],
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Determine if fine-tuned model is meaningfully better than baseline."""
        comparison = self.compare_models(
            baseline_results,
            fine_tuned_results,
            model_a_name="baseline",
            model_b_name="fine_tuned",
        )
        
        # Calculate net improvement
        n_improvements = len(comparison["improvements"])
        n_regressions = len(comparison["regressions"])
        
        # Check if key metrics improved
        key_metrics = ["pass_rate", "accuracy", "safety_score"]
        key_improved = sum(
            1 for imp in comparison["improvements"]
            if imp["metric"] in key_metrics
        )
        
        result = {
            "is_better": comparison["overall_assessment"] == "improvement",
            "net_change": n_improvements - n_regressions,
            "key_metrics_improved": key_improved,
            "total_key_metrics": len(key_metrics),
            "comparison": comparison,
            "recommendation": self._generate_recommendation(comparison, threshold),
        }
        
        return result
    
    def _generate_recommendation(
        self,
        comparison: Dict[str, Any],
        threshold: float,
    ) -> str:
        """Generate recommendation based on comparison."""
        assessment = comparison["overall_assessment"]
        
        if assessment == "improvement":
            return "Fine-tuned model shows improvement over baseline. Recommended for deployment."
        elif assessment == "regression":
            return "Fine-tuned model shows regression. Do not deploy; review training configuration."
        else:
            return "Performance is comparable to baseline. Consider additional training or different hyperparameters."