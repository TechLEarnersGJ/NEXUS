"""
Tests for NEXUS sandboxed code execution.
"""

from nexus.evaluation.sandbox import SandboxExecutor, CodeExecutor, SandboxError


def test_sandbox_basic_execution():
    """Test basic code execution in sandbox."""
    executor = SandboxExecutor()
    result = executor.execute('x = 1 + 1\n_result = x')
    assert result['success'] is True
    assert result['result'] == 2


def test_sandbox_print_output():
    """Test that print output is captured."""
    executor = SandboxExecutor()
    result = executor.execute('print("Hello, World!")')
    assert result['success'] is True
    assert 'Hello, World!' in result['stdout']


def test_sandbox_blocked_import():
    """Test that blocked modules cannot be imported."""
    executor = SandboxExecutor()
    result = executor.execute('import os')
    assert result['success'] is False
    assert 'blocked module' in result['error'].lower()


def test_sandbox_blocked_open():
    """Test that open() is blocked."""
    executor = SandboxExecutor()
    result = executor.execute('open("/etc/passwd")')
    assert result['success'] is False
    assert 'blocked' in result['error'].lower()


def test_sandbox_allowed_module():
    """Test that allowed modules can be imported."""
    executor = SandboxExecutor()
    result = executor.execute('import json\n_result = json.dumps({"a": 1})')
    assert result['success'] is True
    assert result['result'] == '{"a": 1}'


def test_sandbox_blocked_socket():
    """Test that network modules are blocked."""
    executor = SandboxExecutor()
    result = executor.execute('import socket')
    assert result['success'] is False


def test_sandbox_syntax_error():
    """Test that syntax errors are caught."""
    executor = SandboxExecutor()
    result = executor.execute('x = ')
    assert result['success'] is False
    assert result['error_type'] == 'ValidationError'


def test_sandbox_code_executor():
    """Test CodeExecutor with function test cases."""
    ce = CodeExecutor()
    result = ce.execute_python_function(
        code='def add(a, b): return a + b',
        function_name='add',
        test_cases=[
            {'input': {'a': 1, 'b': 2}, 'expected_output': 3},
            {'input': {'a': 5, 'b': 10}, 'expected_output': 15},
        ],
    )
    assert result['success'] is True
    assert result['passed'] == 2
    assert result['failed'] == 0
    assert result['total'] == 2


def test_sandbox_code_executor_failing():
    """Test CodeExecutor with failing test cases."""
    ce = CodeExecutor()
    result = ce.execute_python_function(
        code='def add(a, b): return a - b',
        function_name='add',
        test_cases=[
            {'input': {'a': 1, 'b': 2}, 'expected_output': 3},
        ],
    )
    assert result['success'] is True
    assert result['passed'] == 0
    assert result['failed'] == 1


def test_sandbox_temp_file():
    """Test temp file operations in sandbox."""
    executor = SandboxExecutor()
    file_path = executor.write_temp_file('test.txt', 'hello world')
    assert file_path.endswith('test.txt')
    
    content = executor.read_temp_file('test.txt')
    assert content == 'hello world'


def test_sandbox_path_traversal():
    """Test that path traversal is blocked."""
    executor = SandboxExecutor()
    try:
        executor.write_temp_file('../../../etc/passwd', 'evil')
        assert False, 'Should have raised SandboxError'
    except SandboxError:
        pass


def test_sandbox_multiple_statements():
    """Test multiple statements in sandbox."""
    executor = SandboxExecutor()
    result = executor.execute('''
x = 10
y = 20
_result = x + y
''')
    assert result['success'] is True
    assert result['result'] == 30


def test_sandbox_error_in_code():
    """Test that runtime errors are caught."""
    executor = SandboxExecutor()
    result = executor.execute('_ = 1 / 0')
    assert result['success'] is False
    assert result['error_type'] == 'ZeroDivisionError'