"""
Code evaluation benchmarks for NEXUS.

Evaluates model performance on code generation tasks including:
- Python code generation
- JavaScript code generation
- Syntax correctness checking
- Unit test execution in sandbox
"""

import json
from typing import Any, Dict, List, Optional

from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger
from nexus.evaluation.sandbox import CodeExecutor


class CodeEvaluator:
    """Evaluates code generation capabilities of models."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.executor = CodeExecutor()
        
        # Python code generation test cases
        self.python_test_cases = [
            {
                "name": "simple_function",
                "prompt": "Write a Python function that adds two numbers.",
                "test_code": "def add(a, b): return a + b",
                "test_cases": [
                    {"input": {"a": 1, "b": 2}, "expected_output": 3},
                    {"input": {"a": -1, "b": 1}, "expected_output": 0},
                    {"input": {"a": 0, "b": 0}, "expected_output": 0},
                ],
            },
            {
                "name": "string_reversal",
                "prompt": "Write a Python function that reverses a string.",
                "test_code": "def reverse_string(s): return s[::-1]",
                "test_cases": [
                    {"input": {"s": "hello"}, "expected_output": "olleh"},
                    {"input": {"s": ""}, "expected_output": ""},
                    {"input": {"s": "a"}, "expected_output": "a"},
                ],
            },
            {
                "name": "factorial",
                "prompt": "Write a Python function that computes factorial of n.",
                "test_code": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                "test_cases": [
                    {"input": {"n": 0}, "expected_output": 1},
                    {"input": {"n": 1}, "expected_output": 1},
                    {"input": {"n": 5}, "expected_output": 120},
                ],
            },
            {
                "name": "palindrome_check",
                "prompt": "Write a Python function that checks if a string is a palindrome.",
                "test_code": "def is_palindrome(s): return s == s[::-1]",
                "test_cases": [
                    {"input": {"s": "racecar"}, "expected_output": True},
                    {"input": {"s": "hello"}, "expected_output": False},
                    {"input": {"s": "madam"}, "expected_output": True},
                ],
            },
            {
                "name": "fibonacci",
                "prompt": "Write a Python function that returns the nth Fibonacci number.",
                "test_code": "def fibonacci(n): a, b = 0, 1; [a, b] = [b, a + b] for _ in range(n); return a",
                "test_cases": [
                    {"input": {"n": 0}, "expected_output": 0},
                    {"input": {"n": 1}, "expected_output": 1},
                    {"input": {"n": 10}, "expected_output": 55},
                ],
            },
        ]
        
        # JavaScript code generation test cases
        self.javascript_test_cases = [
            {
                "name": "js_add",
                "prompt": "Write a JavaScript function that adds two numbers.",
                "test_code": "function add(a, b) { return a + b; }",
                "test_cases": [
                    {"input": {"a": 1, "b": 2}, "expected_output": 3},
                ],
            },
            {
                "name": "js_string_length",
                "prompt": "Write a JavaScript function that returns string length.",
                "test_code": "function strLen(s) { return s.length; }",
                "test_cases": [
                    {"input": {"s": "hello"}, "expected_output": 5},
                ],
            },
        ]
    
    def evaluate_python_code_generation(
        self,
        model_id: str,
        generated_code: str,
        test_case_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate Python code generation against test cases."""
        self.nexus_logger.log_event(
            "python_code_eval_start",
            {
                "model_id": model_id,
                "test_case": test_case_name,
                "code_length": len(generated_code),
            }
        )
        
        results = []
        
        for tc in self.python_test_cases:
            if test_case_name and tc["name"] != test_case_name:
                continue
            
            # Execute the generated code with test cases
            eval_result = self.executor.execute_python_function(
                code=generated_code,
                function_name=tc["test_code"].split("def ")[1].split("(")[0]
                if "def " in tc["test_code"]
                else "unknown",
                test_cases=tc["test_cases"],
            )
            
            # Check syntax validity
            syntax_valid = self._check_python_syntax(generated_code)
            
            test_result = {
                "test_name": tc["name"],
                "prompt": tc["prompt"],
                "passed": eval_result["passed"],
                "failed": eval_result["failed"],
                "errors": eval_result["errors"],
                "total": eval_result["total"],
                "pass_rate": eval_result["passed"] / max(eval_result["total"], 1),
                "syntax_valid": syntax_valid,
                "stdout": eval_result.get("stdout", ""),
                "stderr": eval_result.get("stderr", ""),
                "error": eval_result.get("error"),
            }
            
            results.append(test_result)
        
        # Calculate overall metrics
        total_passed = sum(r["passed"] for r in results)
        total_failed = sum(r["failed"] for r in results)
        total_errors = sum(r["errors"] for r in results)
        total_tests = sum(r["total"] for r in results)
        syntax_valid_count = sum(1 for r in results if r["syntax_valid"])
        
        overall = {
            "model_id": model_id,
            "language": "python",
            "total_tests": total_tests,
            "tests_passed": total_passed,
            "tests_failed": total_failed,
            "tests_errored": total_errors,
            "pass_rate": total_passed / max(total_tests, 1),
            "syntax_valid_rate": syntax_valid_count / max(len(results), 1),
            "test_results": results,
        }
        
        self.nexus_logger.log_event(
            "python_code_eval_completed",
            {
                "model_id": model_id,
                "pass_rate": overall["pass_rate"],
                "syntax_valid_rate": overall["syntax_valid_rate"],
            }
        )
        
        return overall
    
    def evaluate_javascript_code_generation(
        self,
        model_id: str,
        generated_code: str,
    ) -> Dict[str, Any]:
        """Evaluate JavaScript code generation.
        
        Note: JavaScript execution is not currently supported in the sandbox.
        This performs syntax pattern checking only.
        """
        self.nexus_logger.log_event(
            "javascript_code_eval_start",
            {
                "model_id": model_id,
                "code_length": len(generated_code),
            }
        )
        
        # Basic JavaScript syntax checks
        js_checks = {
            "has_function_keyword": "function" in generated_code,
            "has_arrow_function": "=>" in generated_code,
            "has_return": "return" in generated_code,
            "has_semicolons": ";" in generated_code,
            "has_const_or_let": "const" in generated_code or "let" in generated_code,
            "balanced_braces": generated_code.count("{") == generated_code.count("}"),
            "balanced_parens": generated_code.count("(") == generated_code.count(")"),
        }
        
        syntax_score = sum(1 for v in js_checks.values() if v) / len(js_checks)
        
        result = {
            "model_id": model_id,
            "language": "javascript",
            "syntax_checks": js_checks,
            "syntax_score": syntax_score,
            "note": "JavaScript execution not supported in sandbox; syntax patterns only",
        }
        
        self.nexus_logger.log_event(
            "javascript_code_eval_completed",
            {
                "model_id": model_id,
                "syntax_score": syntax_score,
            }
        )
        
        return result
    
    def evaluate_syntax_correctness(
        self,
        code: str,
        language: str = "python",
    ) -> Dict[str, Any]:
        """Check syntax correctness of generated code."""
        if language == "python":
            return self._check_python_syntax_detailed(code)
        elif language == "javascript":
            return self._check_javascript_syntax(code)
        else:
            return {
                "language": language,
                "syntax_valid": None,
                "error": f"Unsupported language: {language}",
            }
    
    def _check_python_syntax(self, code: str) -> bool:
        """Check if Python code has valid syntax."""
        try:
            compile(code, '<sandbox>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def _check_python_syntax_detailed(self, code: str) -> Dict[str, Any]:
        """Detailed Python syntax check."""
        try:
            compile(code, '<sandbox>', 'exec')
            return {
                "language": "python",
                "syntax_valid": True,
                "error": None,
            }
        except SyntaxError as e:
            return {
                "language": "python",
                "syntax_valid": False,
                "error": str(e),
                "line": e.lineno,
                "offset": e.offset,
            }
    
    def _check_javascript_syntax(self, code: str) -> Dict[str, Any]:
        """Basic JavaScript syntax check (pattern-based)."""
        checks = {
            "balanced_braces": code.count("{") == code.count("}"),
            "balanced_parens": code.count("(") == code.count(")"),
            "balanced_brackets": code.count("[") == code.count("]"),
            "no_obvious_errors": "function" in code or "=>" in code or "const" in code or "let" in code or "var" in code,
        }
        
        syntax_valid = all(checks.values())
        
        return {
            "language": "javascript",
            "syntax_valid": syntax_valid,
            "checks": checks,
            "error": None if syntax_valid else "JavaScript syntax patterns invalid",
        }
    
    def evaluate_unit_test_execution(
        self,
        model_id: str,
        code: str,
        test_code: str,
    ) -> Dict[str, Any]:
        """Evaluate code by executing unit tests in sandbox."""
        self.nexus_logger.log_event(
            "unit_test_eval_start",
            {
                "model_id": model_id,
                "code_length": len(code),
                "test_code_length": len(test_code),
            }
        )
        
        result = self.executor.sandbox.execute_with_tests(code, test_code)
        
        eval_result = {
            "model_id": model_id,
            "code_executed": result.get("code_result", {}).get("success", False),
            "tests_executed": result.get("test_result", {}).get("success", False),
            "tests_passed": result.get("tests_passed", 0),
            "tests_failed": result.get("tests_failed", 0),
            "total_tests": result.get("total_tests", 0),
            "pass_rate": result.get("tests_passed", 0) / max(result.get("total_tests", 1), 1),
            "code_stdout": result.get("code_result", {}).get("stdout", ""),
            "code_stderr": result.get("code_result", {}).get("stderr", ""),
            "test_stdout": result.get("test_result", {}).get("stdout", ""),
            "test_stderr": result.get("test_result", {}).get("stderr", ""),
            "code_error": result.get("code_result", {}).get("error"),
            "test_error": result.get("test_result", {}).get("error"),
        }
        
        self.nexus_logger.log_event(
            "unit_test_eval_completed",
            {
                "model_id": model_id,
                "pass_rate": eval_result["pass_rate"],
                "tests_passed": eval_result["tests_passed"],
                "tests_failed": eval_result["tests_failed"],
            }
        )
        
        return eval_result
    
    def evaluate_code_complexity(
        self,
        code: str,
        language: str = "python",
    ) -> Dict[str, Any]:
        """Evaluate code complexity metrics."""
        if language != "python":
            return {
                "language": language,
                "error": "Complexity metrics only supported for Python",
            }
        
        try:
            import ast
            tree = ast.parse(code)
        except SyntaxError:
            return {
                "language": "python",
                "error": "Invalid Python syntax",
                "complexity_score": 0,
            }
        
        # Count various constructs
        function_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        if_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.If))
        for_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.For))
        while_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.While))
        try_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Try))
        
        # Simple complexity score
        complexity_score = (
            function_count * 1 +
            class_count * 2 +
            if_count * 1 +
            for_count * 1 +
            while_count * 1 +
            try_count * 1
        )
        
        return {
            "language": "python",
            "function_count": function_count,
            "class_count": class_count,
            "if_count": if_count,
            "for_count": for_count,
            "while_count": while_count,
            "try_count": try_count,
            "complexity_score": complexity_score,
            "lines_of_code": len(code.splitlines()),
        }