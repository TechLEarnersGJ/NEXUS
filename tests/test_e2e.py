"""
End-to-end integration tests for NEXUS.

These tests verify the complete workflow from intent parsing to model export,
using mock providers and simulated data.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from nexus.core.config import Config
from nexus.core.engine import Engine
from nexus.core.state import JobManager
from nexus.orchestration.approval import ApprovalGate
from nexus.orchestration.job_manager import OrchestrationJobManager
from nexus.orchestration.planner import WorkflowPlanner
from nexus.providers.manager import ProviderManager


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test isolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = Config()
        config.load_from_file("config/settings.yaml", "settings")
        config.load_from_file("config/providers.yaml", "providers")
        config.load_from_file("config/training.yaml", "training")
        return config

    @pytest.fixture
    def engine(self, config):
        """Create a test engine."""
        return Engine(config)

    @pytest.fixture
    def job_manager(self, temp_dir):
        """Create a job manager with temporary storage."""
        state_manager = JobManager(storage_dir=str(temp_dir / "jobs"))
        return OrchestrationJobManager(state_manager), temp_dir

    def test_full_workflow_create_plan(self, engine, job_manager):
        """Test the complete create -> plan workflow."""
        om, temp_dir = job_manager

        # Step 1: Create a job
        job_id = om.create_job(
            name="Test Coding Assistant",
            description="Test job for E2E workflow",
            provider="kaggle",
            config={"model": "Qwen2.5-0.5B-Instruct"},
            priority=1,
        )

        assert job_id is not None
        assert job_id.startswith("job_")

        # Step 2: Verify job was created
        job = om.get_job(job_id)
        assert job is not None
        assert job["name"] == "Test Coding Assistant"
        assert job["provider"] == "kaggle"

        # Step 3: Plan a workflow
        planner = WorkflowPlanner()
        model_requirements = {
            "objective": "Create a coding assistant",
            "max_parameters": "1B",
            "languages": ["python", "javascript"],
            "capabilities": {"conversational": True, "code_generation": True},
            "training_method": "qlora",
            "preferred_providers": ["kaggle", "colab"],
            "model_registry": "huggingface",
            "budget": "free",
        }
        dataset_requirements = {
            "objective": "Create a coding assistant",
            "types": ["instruction", "code"],
            "languages": ["python"],
        }

        plan = planner.plan(
            objective="Create a coding assistant",
            model_requirements=model_requirements,
            dataset_requirements=dataset_requirements,
        )

        assert "plan_id" in plan
        assert "steps" in plan
        assert len(plan["steps"]) > 0
        assert "estimated_duration_minutes" in plan
        assert "estimated_gpu_hours" in plan

        # Step 4: Update job with plan
        om.update_job(
            job_id=job_id,
            status="planned",
            progress=0.1,
            metadata_updates={"plan": plan},
        )

        updated_job = om.get_job(job_id)
        assert updated_job["status"] == "planned"
        assert updated_job["progress"] == 0.1
        assert "plan" in updated_job
        assert updated_job["plan"]["plan_id"] is not None

    def test_approval_gate_workflow(self):
        """Test the approval gate workflow for publishing."""
        gate = ApprovalGate()

        # Step 1: Check that publishing requires approval
        assert gate.requires_approval("publish_model") is True

        # Step 2: Check operation without request
        result = gate.check_operation_allowed("publish_model")
        assert result["allowed"] is False
        assert result["requires_approval"] is True

        # Step 3: Create approval request
        result = gate.check_operation_allowed(
            "publish_model",
            auto_create_request=True,
            model_id="test/model",
        )
        assert result["allowed"] is False
        assert result["request_id"] is not None

        # Step 4: Approve the request
        success = gate.approve_request(result["request_id"], "test_user")
        assert success is True

        # Step 5: Check operation after approval
        result = gate.check_operation_allowed("publish_model")
        assert result["allowed"] is True

        # Step 6: Verify history
        history = gate.get_approval_history()
        assert len(history) == 1
        assert history[0]["status"] == "approved"
        assert history[0]["approver"] == "test_user"

    def test_provider_selection(self, config):
        """Test provider selection through manager."""
        manager = ProviderManager(config)

        # Get provider names
        names = manager.get_provider_names()
        assert "kaggle" in names
        assert "colab" in names
        assert "huggingface" in names

        # Authenticate
        results = manager.authenticate()
        assert "kaggle" in results
        assert "colab" in results
        assert "huggingface" in results

    def test_model_discovery_and_selection(self, engine):
        """Test model discovery and selection workflow."""
        # Discover models
        models = engine.discover_models(query="python", max_models=5)
        assert len(models) > 0

        # Check model properties
        for model in models:
            assert model.model_id is not None
            assert model.name is not None
            assert model.parameter_count > 0

        # Select best model
        from nexus.models.selector import ModelSelector
        selector = ModelSelector()

        requirements = {
            "objective": "python coding assistant",
            "max_parameters": 1000000000,  # 1B
            "required_license": "apache-2.0",
            "capabilities": ["instruction_following"],
        }

        selected = selector.select_best_model(models, requirements)
        # May be None if no model matches all criteria
        if selected is not None:
            assert selected.parameter_count <= 1000000000

    def test_dataset_discovery(self, engine):
        """Test dataset discovery workflow."""
        datasets = engine.discover_datasets(query="python", max_datasets=5)
        assert len(datasets) > 0

        for dataset in datasets:
            assert dataset.dataset_id is not None
            assert dataset.name is not None
            assert dataset.license is not None

    def test_training_preflight(self, engine):
        """Test training preflight checks."""
        from nexus.training.preflight import PreflightChecker

        checker = PreflightChecker()

        # Test feasible case
        result = checker.check_feasibility(
            model_parameter_count=500000000,  # 0.5B
            gpu_type="Tesla T4",
            method="qlora",
        )
        assert result.feasible is True
        assert result.recommended_method == "qlora"

        # Test infeasible case
        result = checker.check_feasibility(
            model_parameter_count=70000000000,  # 70B
            gpu_type="Tesla T4",
            method="qlora",
        )
        assert result.feasible is False
        assert len(result.errors) > 0

    def test_job_lifecycle(self, job_manager):
        """Test complete job lifecycle."""
        om, temp_dir = job_manager

        # Create
        job_id = om.create_job(
            name="Lifecycle Test",
            provider="kaggle",
            config={"test": True},
        )

        # Read
        job = om.get_job(job_id)
        assert job is not None
        assert job["status"] == "pending"

        # Update
        om.update_job(job_id=job_id, status="running", progress=0.5)
        job = om.get_job(job_id)
        assert job["status"] == "running"
        assert job["progress"] == 0.5

        # List
        jobs = om.list_jobs()
        assert len(jobs) == 1

        # Get events
        events = om.get_job_events(job_id)
        assert len(events) >= 1

        # Delete
        success = om.delete_job(job_id)
        assert success is True
        assert om.get_job(job_id) is None

    def test_statistics(self, job_manager):
        """Test job statistics."""
        om, temp_dir = job_manager

        om.create_job(name="Job 1", provider="kaggle")
        om.create_job(name="Job 2", provider="kaggle")
        om.create_job(name="Job 3", provider="colab")

        stats = om.get_job_statistics()
        assert stats["total"] == 3
        assert stats["by_status"]["pending"] == 3
        assert stats["by_provider"]["kaggle"] == 2
        assert stats["by_provider"]["colab"] == 1

    def test_evaluation_integration(self, engine):
        """Test evaluation workflow integration."""
        from nexus.evaluation.code_eval import CodeEvaluator
        from nexus.evaluation.safety_eval import SafetyEvaluator
        from nexus.evaluation.regression import RegressionTester

        # Code evaluation
        ce = CodeEvaluator()
        result = ce.evaluate_python_code_generation(
            model_id="test-model",
            generated_code="def add(a, b): return a + b",
            test_case_name="simple_function",
        )
        assert result["pass_rate"] == 1.0

        # Safety evaluation
        se = SafetyEvaluator()
        result = se.evaluate_safety(
            model_id="test-model",
            response="I cannot help with that request.",
        )
        assert result["is_safe"] is True

        # Regression testing
        rt = RegressionTester()
        baseline = {"pass_rate": 0.6, "accuracy": 0.7}
        fine_tuned = {"pass_rate": 0.85, "accuracy": 0.75}
        result = rt.evaluate_improvement(baseline, fine_tuned)
        assert result["is_better"] is True

    def test_report_generation(self, temp_dir):
        """Test evaluation report generation."""
        from nexus.evaluation.reports import EvaluationReporter

        reporter = EvaluationReporter(output_dir=str(temp_dir / "reports"))

        result = reporter.generate_training_report(
            training_results={
                "status": "completed",
                "model_id": "test-model",
                "dataset_id": "test-dataset",
                "epochs_completed": 3,
                "final_loss": 0.45,
                "training_time_seconds": 120.5,
            },
            evaluation_before={"pass_rate": 0.6, "accuracy": 0.7},
            evaluation_after={"pass_rate": 0.85, "accuracy": 0.75},
            report_name="test_report",
        )

        assert "json" in result
        assert "markdown" in result
        assert Path(result["json"]).exists()
        assert Path(result["markdown"]).exists()

        # Verify report content
        import json
        with open(result["json"], "r") as f:
            report = json.load(f)
        assert report["report_name"] == "test_report"
        assert report["training_results"]["status"] == "completed"

    def test_sandbox_security(self):
        """Test sandbox security restrictions."""
        from nexus.evaluation.sandbox import SandboxExecutor

        executor = SandboxExecutor()

        # Test blocked imports
        result = executor.execute("import os")
        assert result["success"] is False

        result = executor.execute("import socket")
        assert result["success"] is False

        # Test blocked builtins
        result = executor.execute("open('/etc/passwd')")
        assert result["success"] is False

        result = executor.execute("eval('1+1')")
        assert result["success"] is False

        # Test allowed operations
        result = executor.execute("x = 1 + 1\n_result = x")
        assert result["success"] is True
        assert result["result"] == 2

        result = executor.execute("import json\n_result = json.dumps({'a': 1})")
        assert result["success"] is True

    def test_notebook_generation(self):
        """Test notebook generation for providers."""
        from nexus.training.notebook_generator import NotebookGenerator

        gen = NotebookGenerator()

        # Colab notebook
        result = gen.generate_colab_notebook(
            job_id="test_job",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            dataset_id="test/dataset",
            training_config={"method": "qlora", "config": {}},
        )
        assert result["provider"] == "colab"
        assert len(result["notebook"]["cells"]) > 0
        assert "instructions" in result

        # Kaggle notebook
        result = gen.generate_kaggle_notebook(
            job_id="test_job",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            dataset_id="test/dataset",
            training_config={"method": "qlora", "config": {}},
        )
        assert result["provider"] == "kaggle"
        assert len(result["notebook"]["cells"]) > 0

    def test_full_e2e_simulation(self, config, job_manager):
        """Simulate a complete end-to-end workflow."""
        om, temp_dir = job_manager
        planner = WorkflowPlanner()

        # 1. Parse intent
        objective_text = (
            "Create a small conversational programming assistant "
            "under 1B parameters, specializing in Python and JavaScript"
        )

        # 2. Create job
        job_id = om.create_job(
            name="Example Coding Assistant",
            description=objective_text,
            provider="kaggle",
            config={"model": "Qwen2.5-0.5B-Instruct"},
        )

        # 3. Plan workflow
        plan = planner.create_reproducible_plan(objective_text)
        assert plan["plan_id"] is not None
        assert len(plan["steps"]) > 0

        # 4. Update job with plan
        om.update_job(
            job_id=job_id,
            status="planned",
            progress=0.1,
            metadata_updates={"plan": plan},
        )

        # 5. Verify final state
        job = om.get_job(job_id)
        assert job["status"] == "planned"
        assert job["progress"] == 0.1

        # 6. Verify job statistics
        stats = om.get_job_statistics()
        assert stats["total"] == 1

        print("\n=== E2E Simulation Complete ===")
        print(f"Job ID: {job_id}")
        print(f"Plan ID: {plan['plan_id']}")
        print(f"Steps: {len(plan['steps'])}")
        print(f"Estimated GPU hours: {plan['estimated_gpu_hours']}")