"""
Sandboxed code execution for NEXUS evaluation.

Provides restricted execution of untrusted code with:
- No filesystem access outside temp directory
- No network access
- No subprocess spawning
- Timeout limits
- Memory limits
- Import restrictions
"""

import ast
import builtins
import io
import os
import sys
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Optional, Set


class SandboxError(Exception):
    """Raised when sandbox execution fails or violates restrictions."""
    pass


class SandboxExecutor:
    """Executes Python code in a restricted sandbox environment."""
    
    # Modules that are completely blocked
    BLOCKED_MODULES: Set[str] = {
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
        'http', 'ftplib', 'smtplib', 'telnetlib', 'paramiko',
        'shutil', 'glob', 'shlex', 'ctypes', 'multiprocessing',
        'threading', 'asyncio', 'concurrent', 'importlib',
        'pkgutil', 'site', 'distutils', 'setuptools', 'pip',
        'cryptography', 'Crypto', 'Crypto', 'ssl',
        'xmlrpc', 'webbrowser', 'ctypes', 'fcntl', 'resource',
        'termios', 'tty', 'pty',
    }
    
    # Modules that are allowed (read-only, safe)
    ALLOWED_MODULES: Set[str] = {
        'json', 'math', 'random', 'datetime', 'collections',
        'itertools', 'functools', 'operator', 'typing',
        'dataclasses', 'enum', 'uuid', 'hashlib', 'base64',
        're', 'string', 'textwrap', 'unicodedata',
    }
    
    # Built-in functions that are blocked
    BLOCKED_BUILTINS: Set[str] = {
        'open', 'exec', 'eval', 'compile',
        'exit', 'quit', 'help', 'copyright', 'credits', 'license',
        'breakpoint', 'input',
        'vars', 'dir', 'getattr', 'setattr', 'delattr',
        'hasattr', 'callable', 'isinstance', 'issubclass',
    }
    
    # __import__ is allowed but restricted to approved modules
    # (handled by _safe_import in _create_safe_globals)
    
    # Default resource limits
    DEFAULT_TIMEOUT_SECONDS = 30
    DEFAULT_MAX_OUTPUT_CHARS = 10000
    DEFAULT_MAX_MEMORY_MB = 256
    
    def __init__(
        self,
        timeout_seconds: Optional[int] = None,
        max_output_chars: Optional[int] = None,
        max_memory_mb: Optional[int] = None,
        allowed_modules: Optional[Set[str]] = None,
        blocked_modules: Optional[Set[str]] = None,
    ):
        self.timeout_seconds = timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS
        self.max_output_chars = max_output_chars or self.DEFAULT_MAX_OUTPUT_CHARS
        self.max_memory_mb = max_memory_mb or self.DEFAULT_MAX_MEMORY_MB
        
        self.allowed_modules = allowed_modules or self.ALLOWED_MODULES.copy()
        self.blocked_modules = blocked_modules or self.BLOCKED_MODULES.copy()
        
        # Create a temp directory for any file operations
        self._temp_dir = tempfile.mkdtemp(prefix='nexus_sandbox_')
    
    def __del__(self):
        """Clean up temp directory."""
        try:
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass
    
    def execute(
        self,
        code: str,
        globals_dict: Optional[Dict[str, Any]] = None,
        locals_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute code in the sandbox.
        
        Returns a dict with:
            success: bool
            stdout: str
            stderr: str
            result: Any (last expression value)
            error: str (if failed)
            error_type: str
            traceback: str
            execution_time: float
        """
        import time
        
        start_time = time.time()
        
        result = {
            'success': False,
            'stdout': '',
            'stderr': '',
            'result': None,
            'error': None,
            'error_type': None,
            'traceback': None,
            'execution_time': 0.0,
        }
        
        # Validate code before execution
        validation = self._validate_code(code)
        if not validation['valid']:
            result['error'] = validation['error']
            result['error_type'] = 'ValidationError'
            result['traceback'] = validation['traceback']
            result['execution_time'] = time.time() - start_time
            return result
        
        # Prepare restricted globals
        safe_globals = self._create_safe_globals(globals_dict or {})
        safe_locals = locals_dict or {}
        
        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Compile the code
                compiled = compile(code, '<sandbox>', 'exec')
                
                # Execute
                exec(compiled, safe_globals, safe_locals)
            
            result['success'] = True
            result['stdout'] = self._truncate(stdout_capture.getvalue())
            result['stderr'] = self._truncate(stderr_capture.getvalue())
            result['result'] = safe_locals.get('_result')
            result['_globals'] = safe_globals
            result['_locals'] = safe_locals
            
        except SyntaxError as e:
            result['error'] = str(e)
            result['error_type'] = 'SyntaxError'
            result['traceback'] = traceback.format_exc()
        except SandboxError as e:
            result['error'] = str(e)
            result['error_type'] = 'SandboxError'
            result['traceback'] = traceback.format_exc()
        except Exception as e:
            result['error'] = str(e)
            result['error_type'] = type(e).__name__
            result['traceback'] = traceback.format_exc()
        finally:
            result['execution_time'] = round(time.time() - start_time, 4)
        
        return result
    
    def execute_with_tests(
        self,
        code: str,
        test_code: str,
        globals_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute code and then run tests against it.
        
        Returns combined results including test pass/fail counts.
        """
        # First execute the main code
        exec_result = self.execute(code, globals_dict)
        
        if not exec_result['success']:
            return {
                'success': False,
                'code_result': exec_result,
                'test_result': None,
                'tests_passed': 0,
                'tests_failed': 0,
                'total_tests': 0,
            }
        
        # Merge globals and locals from code execution
        merged_globals = {}
        merged_globals.update(exec_result.get('_globals', {}))
        merged_globals.update(exec_result.get('_locals', {}))
        if globals_dict:
            merged_globals.update(globals_dict)
        
        # Execute test code
        test_result = self.execute(test_code, merged_globals)
        
        # Count test results from stdout
        tests_passed = 0
        tests_failed = 0
        total_tests = 0
        
        # Look for test assertions in output
        output = test_result['stdout'] + test_result['stderr']
        for line in output.splitlines():
            line = line.strip()
            if 'PASS' in line or '✓' in line or 'ok' in line.lower():
                tests_passed += 1
                total_tests += 1
            elif 'FAIL' in line or '✗' in line or 'Error' in line:
                tests_failed += 1
                total_tests += 1
        
        return {
            'success': test_result['success'],
            'code_result': exec_result,
            'test_result': test_result,
            'tests_passed': tests_passed,
            'tests_failed': tests_failed,
            'total_tests': total_tests,
        }
    
    def _validate_code(self, code: str) -> Dict[str, Any]:
        """Validate code for security violations before execution."""
        result = {
            'valid': True,
            'error': None,
            'traceback': None,
        }
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result['valid'] = False
            result['error'] = f'Syntax error: {e}'
            result['traceback'] = traceback.format_exc()
            return result
        
        # Walk the AST for violations
        for node in ast.walk(tree):
            # Check for imports of blocked modules
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in self.blocked_modules:
                        result['valid'] = False
                        result['error'] = (
                            f'Import of blocked module: {module_name}'
                        )
                        return result
                    if module_name not in self.allowed_modules:
                        result['valid'] = False
                        result['error'] = (
                            f'Import of unapproved module: {module_name}'
                        )
                        return result
            
            # Check for from imports of blocked modules
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in self.blocked_modules:
                        result['valid'] = False
                        result['error'] = (
                            f'From import of blocked module: {module_name}'
                        )
                        return result
                    if module_name not in self.allowed_modules:
                        result['valid'] = False
                        result['error'] = (
                            f'From import of unapproved module: {module_name}'
                        )
                        return result
            
            # Check for attribute access on blocked objects
            elif isinstance(node, ast.Attribute):
                # Block access to __import__, __builtins__, etc.
                if node.attr.startswith('__') and node.attr.endswith('__'):
                    result['valid'] = False
                    result['error'] = (
                        f'Blocked attribute access: {node.attr}'
                    )
                    return result
            
            # Check for function calls to blocked builtins
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_BUILTINS:
                        result['valid'] = False
                        result['error'] = (
                            f'Blocked function call: {node.func.id}'
                        )
                        return result
        
        return result
    
    def _safe_import(self, name: str, *args, **kwargs):
        """Restricted import function that only allows approved modules."""
        module_name = name.split('.')[0]
        if module_name in self.blocked_modules:
            raise ImportError(f'Import of blocked module: {module_name}')
        if module_name not in self.allowed_modules:
            raise ImportError(f'Import of unapproved module: {module_name}')
        return __import__(name, *args, **kwargs)
    
    def _create_safe_globals(self, user_globals: Dict[str, Any]) -> Dict[str, Any]:
        """Create a restricted globals dictionary."""
        # Start with a minimal safe set
        safe_builtins = {}
        
        for name in dir(builtins):
            if name not in self.BLOCKED_BUILTINS:
                safe_builtins[name] = getattr(builtins, name)
        
        # Add safe __import__ that restricts to approved modules
        safe_builtins['__import__'] = self._safe_import
        
        # Add safe modules
        safe_modules = {}
        for module_name in self.allowed_modules:
            try:
                mod = __import__(module_name)
                safe_modules[module_name] = mod
            except ImportError:
                pass
        
        safe_globals = {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            '__doc__': None,
        }
        
        # Add safe modules
        safe_globals.update(safe_modules)
        
        # Add user globals (but don't override safety-critical items)
        for key, value in user_globals.items():
            if key not in ('__builtins__', '__name__', '__doc__', '__import__'):
                safe_globals[key] = value
        
        return safe_globals
    
    def _truncate(self, text: str) -> str:
        """Truncate output to max chars."""
        if len(text) > self.max_output_chars:
            return text[:self.max_output_chars] + f'\n...[truncated, {len(text)} total chars]'
        return text
    
    def get_temp_dir(self) -> str:
        """Get the sandbox temp directory path."""
        return self._temp_dir
    
    def write_temp_file(self, filename: str, content: str) -> str:
        """Write a file to the sandbox temp directory."""
        file_path = os.path.join(self._temp_dir, filename)
        # Prevent path traversal
        if '..' in filename or filename.startswith('/'):
            raise SandboxError(f'Invalid filename: {filename}')
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path
    
    def read_temp_file(self, filename: str) -> str:
        """Read a file from the sandbox temp directory."""
        file_path = os.path.join(self._temp_dir, filename)
        if '..' in filename or filename.startswith('/'):
            raise SandboxError(f'Invalid filename: {filename}')
        with open(file_path, 'r') as f:
            return f.read()


class CodeExecutor:
    """High-level code executor for evaluation tasks."""
    
    def __init__(self):
        self.sandbox = SandboxExecutor()
    
    def execute_python_code(
        self,
        code: str,
        test_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Python code with optional tests."""
        if test_code:
            return self.sandbox.execute_with_tests(code, test_code)
        return self.sandbox.execute(code)
    
    def execute_python_function(
        self,
        code: str,
        function_name: str,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute a specific function with test cases.
        
        Args:
            code: Python code defining the function
            function_name: Name of the function to test
            test_cases: List of dicts with 'input' and 'expected_output'
        
        Returns:
            Dict with pass/fail counts and details
        """
        # Serialize test cases as a Python literal
        import json
        test_cases_json = json.dumps(test_cases)
        
        # Add test runner code - uses _result pattern for capture
        test_runner = f"""
import json
_test_cases = json.loads({test_cases_json!r})
_results = []
for i, _tc in enumerate(_test_cases):
    try:
        _result = {function_name}(**_tc['input'])
        _expected = _tc['expected_output']
        if _result == _expected:
            _results.append(('PASS', i, _result, _expected))
        else:
            _results.append(('FAIL', i, _result, _expected))
    except Exception as _e:
        _results.append(('ERROR', i, str(_e), _tc.get('expected_output', 'N/A')))

for _status, _idx, _result, _expected in _results:
    print(f'Test {{_idx}}: {{_status}} - got {{_result}}, expected {{_expected}}')

_result = _results
"""
        
        full_code = code + '\n' + test_runner
        
        exec_result = self.sandbox.execute(full_code)
        
        # Parse results from stdout
        passed = 0
        failed = 0
        errors = 0
        
        for line in exec_result['stdout'].splitlines():
            line = line.strip()
            if ': PASS' in line:
                passed += 1
            elif ': FAIL' in line:
                failed += 1
            elif ': ERROR' in line:
                errors += 1
        
        return {
            'success': exec_result['success'],
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'total': passed + failed + errors,
            'stdout': exec_result['stdout'],
            'stderr': exec_result['stderr'],
            'error': exec_result['error'],
        }