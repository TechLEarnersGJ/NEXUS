"""
Tests for NEXUS evaluation benchmarks.
"""

from nexus.evaluation.benchmarks import BenchmarkSuite


def test_benchmark_suite_initialization():
    """Test benchmark suite initialization."""
    suite = BenchmarkSuite()
    
    assert len(suite.benchmarks) == 5
    assert "code_generation" in suite.benchmarks
    assert "chat_quality" in suite.benchmarks
    assert "safety" in suite.benchmarks
    assert "code_correctness" in suite.benchmarks
    assert "performance" in suite.benchmarks


def test_benchmark_run_code_generation():
    """Test code generation benchmark."""
    suite = BenchmarkSuite()
    
    result = suite.run_benchmark("code_generation", "test-model")
    
    assert result["benchmark_name"] == "code_generation"
    assert result["model_id"] == "test-model"
    assert result["success"] is True
    assert 0.0 <= result["score"] <= 1.0
    assert "metrics" in result
    assert "test_samples" in result


def test_benchmark_run_chat_quality():
    """Test chat quality benchmark."""
    suite = BenchmarkSuite()
    
    result = suite.run_benchmark("chat_quality", "test-model")
    
    assert result["benchmark_name"] == "chat_quality"
    assert result["success"] is True
    assert 0.0 <= result["score"] <= 1.0


def test_benchmark_run_safety():
    """Test safety benchmark."""
    suite = BenchmarkSuite()
    
    result = suite.run_benchmark("safety", "test-model")
    
    assert result["benchmark_name"] == "safety"
    assert result["success"] is True
    assert 0.0 <= result["score"] <= 1.0


def test_benchmark_run_all():
    """Test running all benchmarks."""
    suite = BenchmarkSuite()
    
    results = suite.run_all_benchmarks("test-model")
    
    assert "results" in results
    assert "overall_score" in results
    assert results["total_benchmarks"] == 5
    assert results["successful_benchmarks"] == 5
    assert 0.0 <= results["overall_score"] <= 1.0


def test_benchmark_invalid_name():
    """Test benchmark with invalid name raises error."""
    suite = BenchmarkSuite()
    
    try:
        suite.run_benchmark("invalid_benchmark", "test-model")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected


def test_benchmark_score_range():
    """Test that all benchmark scores are in valid range."""
    suite = BenchmarkSuite()
    
    results = suite.run_all_benchmarks("test-model")
    
    for name, result in results["results"].items():
        if result.get("success"):
            assert 0.0 <= result["score"] <= 1.0, f"Benchmark {name} has invalid score: {result['score']}"