"""
Evaluation reports for NEXUS.

Generates Markdown and JSON reports summarizing evaluation results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


class EvaluationReporter:
    """Generates evaluation reports in various formats."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.output_dir = Path(output_dir or "./workspace/outputs/evaluations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        evaluation_results: Dict[str, Any],
        report_name: Optional[str] = None,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Generate evaluation report(s).
        
        Args:
            evaluation_results: Results from evaluation
            report_name: Name for the report (auto-generated if None)
            formats: List of formats ('json', 'markdown')
        
        Returns:
            Dict mapping format to output file path
        """
        formats = formats or ["json", "markdown"]
        report_name = report_name or f"evaluation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        output_files = {}
        
        if "json" in formats:
            output_files["json"] = self._write_json_report(
                evaluation_results, report_name
            )
        
        if "markdown" in formats:
            output_files["markdown"] = self._write_markdown_report(
                evaluation_results, report_name
            )
        
        self.nexus_logger.log_event(
            "report_generated",
            {
                "report_name": report_name,
                "formats": list(output_files.keys()),
            }
        )
        
        return output_files
    
    def generate_training_report(
        self,
        training_results: Dict[str, Any],
        evaluation_before: Optional[Dict[str, Any]] = None,
        evaluation_after: Optional[Dict[str, Any]] = None,
        report_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate a comprehensive training report.
        
        Includes training metrics, before/after evaluation comparison,
        and regression analysis.
        """
        report_name = report_name or f"training_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Build comprehensive report data
        report_data = {
            "report_name": report_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "training_results": training_results,
            "evaluation_before": evaluation_before,
            "evaluation_after": evaluation_after,
        }
        
        # Add regression comparison if both evaluations available
        if evaluation_before and evaluation_after:
            from nexus.evaluation.regression import RegressionTester
            tester = RegressionTester()
            report_data["regression_analysis"] = tester.evaluate_improvement(
                evaluation_before,
                evaluation_after,
            )
        
        return self.generate_report(report_data, report_name, formats=["json", "markdown"])
    
    def _write_json_report(
        self,
        data: Dict[str, Any],
        report_name: str,
    ) -> str:
        """Write JSON report."""
        output_path = self.output_dir / f"{report_name}.json"
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"JSON report written: {output_path}")
        return str(output_path)
    
    def _write_markdown_report(
        self,
        data: Dict[str, Any],
        report_name: str,
    ) -> str:
        """Write Markdown report."""
        output_path = self.output_dir / f"{report_name}.md"
        
        lines = []
        lines.append(f"# {data.get('report_name', 'Evaluation Report')}")
        lines.append("")
        lines.append(f"**Generated:** {data.get('generated_at', 'N/A')}")
        lines.append("")
        
        # Training results section
        if "training_results" in data:
            lines.append("## Training Results")
            lines.append("")
            tr = data["training_results"]
            lines.append(f"- **Status:** {tr.get('status', 'N/A')}")
            lines.append(f"- **Model:** {tr.get('model_id', 'N/A')}")
            lines.append(f"- **Dataset:** {tr.get('dataset_id', 'N/A')}")
            lines.append(f"- **Epochs:** {tr.get('epochs_completed', 'N/A')}")
            lines.append(f"- **Final Loss:** {tr.get('final_loss', 'N/A')}")
            lines.append(f"- **Training Time:** {tr.get('training_time_seconds', 'N/A'):.2f}s")
            lines.append("")
        
        # Evaluation sections
        for key in ["evaluation_before", "evaluation_after"]:
            if key in data and data[key]:
                section_title = key.replace("_", " ").title()
                lines.append(f"## {section_title}")
                lines.append("")
                lines.append(self._format_eval_markdown(data[key]))
                lines.append("")
        
        # Regression analysis
        if "regression_analysis" in data:
            lines.append("## Regression Analysis")
            lines.append("")
            ra = data["regression_analysis"]
            lines.append(f"- **Is Better:** {ra.get('is_better', False)}")
            lines.append(f"- **Net Change:** {ra.get('net_change', 0)}")
            lines.append(f"- **Recommendation:** {ra.get('recommendation', 'N/A')}")
            lines.append("")
            
            comp = ra.get("comparison", {})
            lines.append(f"- **Overall Assessment:** {comp.get('overall_assessment', 'N/A')}")
            lines.append(f"- **Improvements:** {len(comp.get('improvements', []))}")
            lines.append(f"- **Regressions:** {len(comp.get('regressions', []))}")
            lines.append("")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        self.logger.info(f"Markdown report written: {output_path}")
        return str(output_path)
    
    def _format_eval_markdown(
        self,
        eval_data: Dict[str, Any],
    ) -> str:
        """Format evaluation data as Markdown."""
        lines = []
        
        # Common fields
        for key in ["model_id", "language", "pass_rate", "accuracy", "safety_score"]:
            if key in eval_data:
                lines.append(f"- **{key.replace('_', ' ').title()}:** {eval_data[key]}")
        
        lines.append("")
        
        # Test results
        if "test_results" in eval_data:
            lines.append("### Test Results")
            lines.append("")
            lines.append("| Test | Passed | Failed | Total | Pass Rate |")
            lines.append("|------|--------|--------|-------|-----------|")
            for tr in eval_data["test_results"]:
                lines.append(
                    f"| {tr.get('test_name', 'N/A')} | "
                    f"{tr.get('passed', 0)} | "
                    f"{tr.get('failed', 0)} | "
                    f"{tr.get('total', 0)} | "
                    f"{tr.get('pass_rate', 0):.2%} |"
                )
            lines.append("")
        
        return "\n".join(lines)
    
    def list_reports(self) -> List[Dict[str, Any]]:
        """List all generated reports."""
        reports = []
        
        if not self.output_dir.exists():
            return reports
        
        for f in sorted(self.output_dir.iterdir()):
            if f.is_file() and f.suffix in (".json", ".md"):
                reports.append({
                    "name": f.name,
                    "path": str(f),
                    "format": f.suffix[1:],
                    "size_bytes": f.stat().st_size,
                    "created_at": f.stat().st_ctime,
                })
        
        return reports