"""
Tests for NEXUS orchestration components.
"""

import tempfile
import time
from pathlib import Path

from nexus.core.config import Config
from nexus.core.state import JobManager as StateJobManager
from nexus.orchestration.approval import ApprovalGate
from nexus.orchestration.job_manager import OrchestrationJobManager
from nexus.orchestration.planner import ExecutionPlan, TrainingObjective, WorkflowPlanner
from nexus.orchestration.scheduler import JobScheduler
from nexus.orchestration.workflow import WorkflowExecutor
from nexus.providers.manager import ProviderManager


def _create_temp_job_manager():
    """Create a JobManager with a temporary storage directory."""
    tmpdir = Path(tempfile.mkdtemp(prefix="nexus_test_"))
    state_manager = StateJobManager(storage_dir=str(tmpdir / "jobs"))
    return OrchestrationJobManager(state_manager), tmpdir


def test_training_objective_creation():
    """Test TrainingObjective creation and serialization."""
    obj = TrainingObjective(
        objective="Create a coding assistant",
        max_parameters="1B",
        languages=["python", "javascript"],
        chat_capability=True,
        code_capability=True,
        training_method="qlora",
    )
    
    assert obj.objective == "Create a coding assistant"
    assert obj.max_parameters == "1B"
    assert obj.languages == ["python", "javascript"]
    assert obj.chat_capability is True
    assert obj.code_capability is True
    assert obj.training_method == "qlora"
    
    # Test serialization
    d = obj.to_dict()
    assert d["objective"] == "Create a coding assistant"
    assert d["max_parameters"] == "1B"


def test_execution_plan_creation():
    """Test ExecutionPlan creation and hash."""
    obj = TrainingObjective(
        objective="Test objective",
        max_parameters="0.5B",
    )
    
    plan = ExecutionPlan(
        plan_id="plan_001",
        objective=obj,
        steps=[{"step_id": "step1", "description": "Test step"}],
        estimated_duration_minutes=60,
        estimated_gpu_hours=2.0,
        provider_sequence=["kaggle"],
        checkpoint_strategy="epoch_sync",
        created_at=time.time(),
    )
    
    assert plan.plan_id == "plan_001"
    assert plan.objective.objective == "Test objective"
    assert len(plan.steps) == 1
    
    # Test hash
    plan_hash = plan.get_plan_hash()
    assert len(plan_hash) == 16
    
    # Same plan should produce same hash
    plan_hash2 = plan.get_plan_hash()
    assert plan_hash == plan_hash2


def test_workflow_planner_parse_intent():
    """Test intent parsing from user request."""
    planner = WorkflowPlanner()
    
    objective = planner.parse_intent(
        "Create a small conversational programming assistant under 1 billion parameters, specializing in Python and JavaScript."
    )
    
    assert "programming assistant" in objective.objective
    assert objective.max_parameters == "1B"
    assert "python" in objective.languages
    assert "javascript" in objective.languages
    assert objective.chat_capability is True
    assert objective.code_capability is True
    assert objective.training_method == "qlora"


def test_workflow_planner_parse_intent_small_model():
    """Test intent parsing for small models."""
    planner = WorkflowPlanner()
    
    objective = planner.parse_intent("Create a model under 500M parameters")
    
    assert objective.max_parameters == "0.5B"


def test_workflow_planner_create_plan():
    """Test complete plan creation."""
    planner = WorkflowPlanner()
    
    model_requirements = {
        "objective": "Create a coding assistant",
        "max_parameters": "1B",
        "languages": ["python", "javascript"],
        "capabilities": {"conversational": True, "code_generation": True},
        "training_method": "qlora",
        "preferred_providers": ["kaggle", "colab"],
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
    assert "provider_sequence" in plan
    
    # Check step structure
    step = plan["steps"][0]
    assert "step_id" in step
    assert "description" in plan["steps"][0]
    assert "status" in step


def test_workflow_planner_create_reproducible_plan():
    """Test reproducible plan creation from user request."""
    planner = WorkflowPlanner()
    
    plan = planner.create_reproducible_plan(
        "Create a small conversational programming assistant under 1B parameters"
    )
    
    assert "plan_id" in plan
    assert "objective" in plan
    assert plan["objective"]["max_parameters"] == "1B"
    
    # Same request should produce identical plan (reproducibility)
    plan2 = planner.create_reproducible_plan(
        "Create a small conversational programming assistant under 1B parameters"
    )
    assert plan["plan_id"] == plan2["plan_id"]  # Same timestamp + same objective
    
    # Verify reproducibility via plan hash (content-based, not timestamp-based)
    from nexus.orchestration.planner import ExecutionPlan
    plan_hash_1 = ExecutionPlan(
        plan_id=plan["plan_id"],
        objective=TrainingObjective(**plan["objective"]),
        steps=plan["steps"],
        estimated_duration_minutes=plan["estimated_duration_minutes"],
        estimated_gpu_hours=plan["estimated_gpu_hours"],
        provider_sequence=plan["provider_sequence"],
        checkpoint_strategy=plan["checkpoint_strategy"],
        created_at=plan["created_at"],
    ).get_plan_hash()
    plan_hash_2 = ExecutionPlan(
        plan_id=plan2["plan_id"],
        objective=TrainingObjective(**plan2["objective"]),
        steps=plan2["steps"],
        estimated_duration_minutes=plan2["estimated_duration_minutes"],
        estimated_gpu_hours=plan2["estimated_gpu_hours"],
        provider_sequence=plan2["provider_sequence"],
        checkpoint_strategy=plan2["checkpoint_strategy"],
        created_at=plan2["created_at"],
    ).get_plan_hash()
    assert plan_hash_1 == plan_hash_2


def test_approval_gate_creation():
    """Test ApprovalGate creation."""
    gate = ApprovalGate()
    
    assert gate.auto_approve is False
    assert "publish_model" in gate.require_approval_for
    assert "use_private_dataset" in gate.require_approval_for


def test_approval_gate_request_and_approve():
    """Test approval request and approval flow."""
    gate = ApprovalGate()
    
    # Check that publishing requires approval
    assert gate.requires_approval("publish_model") is True
    assert gate.requires_approval("training_job_submit") is False
    
    # Request approval
    request = gate.request_approval(
        operation_type="publish_model",
        description="Publish fine-tuned model to Hugging Face",
        details={"model_id": "test/model", "repo": "user/model"},
    )
    
    assert request.status == "pending"
    assert request.request_id.startswith("approval_")
    
    # Get pending requests
    pending = gate.get_pending_requests()
    assert len(pending) == 1
    
    # Approve the request
    success = gate.approve_request(request.request_id, "user1")
    assert success is True
    
    # Check status
    updated_request = gate.get_request(request.request_id)
    assert updated_request.status == "approved"
    assert updated_request.approver == "user1"


def test_approval_gate_reject():
    """Test approval rejection."""
    gate = ApprovalGate()
    
    request = gate.request_approval(
        operation_type="publish_model",
        description="Test publish",
    )
    
    # Reject
    success = gate.reject_request(request.request_id, "user1", "Not enough testing")
    assert success is True
    
    updated_request = gate.get_request(request.request_id)
    assert updated_request.status == "rejected"
    assert updated_request.reason == "Not enough testing"


def test_approval_gate_check_operation():
    """Test check_operation_allowed."""
    gate = ApprovalGate()
    
    # Operation not requiring approval
    result = gate.check_operation_allowed("training_job_submit")
    assert result["allowed"] is True
    assert result["requires_approval"] is False
    
    # Operation requiring approval, no pending request
    result = gate.check_operation_allowed("publish_model")
    assert result["allowed"] is False
    assert result["requires_approval"] is True
    assert result["request_id"] is None
    
    # Create a request and check again
    result = gate.check_operation_allowed("publish_model", auto_create_request=True)
    assert result["allowed"] is False
    assert result["requires_approval"] is True
    assert result["request_id"] is not None
    
    # Approve and check
    gate.approve_request(result["request_id"], "user1")
    result = gate.check_operation_allowed("publish_model")
    assert result["allowed"] is True


def test_approval_gate_auto_approve():
    """Test auto-approve mode."""
    gate = ApprovalGate(auto_approve=True)
    
    request = gate.request_approval(
        operation_type="publish_model",
        description="Auto-approve test",
    )
    
    assert request.status == "approved"


def test_execution_plan_hash_deterministic():
    """Test that plan hash is deterministic for same content."""
    import time as time_module
    
    obj1 = TrainingObjective(objective="Test", max_parameters="1B")
    obj2 = TrainingObjective(objective="Test", max_parameters="1B")
    
    plan1 = ExecutionPlan(
        plan_id="plan_1",
        objective=obj1,
        steps=[{"step_id": "s1"}],
        estimated_duration_minutes=60,
        estimated_gpu_hours=2.0,
        provider_sequence=["kaggle"],
        checkpoint_strategy="sync",
        created_at=time_module.time(),
    )
    
    plan2 = ExecutionPlan(
        plan_id="plan_2",
        objective=obj2,
        steps=[{"step_id": "s1"}],
        estimated_duration_minutes=60,
        estimated_gpu_hours=2.0,
        provider_sequence=["kaggle"],
        checkpoint_strategy="sync",
        created_at=time_module.time(),
    )
    
    # Same content should produce same hash
    assert plan1.get_plan_hash() == plan2.get_plan_hash()


def test_scheduler_creation():
    """Test JobScheduler creation."""
    from nexus.core.config import Config
    from nexus.providers.manager import ProviderManager
    
    config = Config()
    manager = ProviderManager(config)
    scheduler = JobScheduler(manager)
    
    assert scheduler.max_concurrent_jobs == 3
    assert scheduler.default_timeout_hours == 24


def test_job_manager_creation():
    """Test OrchestrationJobManager creation."""
    from nexus.core.state import JobManager as StateJobManager
    
    state_manager = StateJobManager()
    manager = OrchestrationJobManager(state_manager)
    
    assert manager.state_job_manager is state_manager


def test_job_manager_create_and_get():
    """Test job creation and retrieval."""
    manager, tmpdir = _create_temp_job_manager()
    
    try:
        # Create job
        job_id = manager.create_job(
            name="Test Job",
            description="A test job",
            provider="kaggle",
            config={"model": "Qwen2.5-0.5B"},
            priority=2,
        )
    
        assert job_id.startswith("job_")
    
        # Get job
        job = manager.get_job(job_id)
        assert job is not None
        assert job["name"] == "Test Job"
        assert job["provider"] == "kaggle"
        assert job["status"] == "pending"
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_job_manager_update():
    """Test job status update."""
    manager, tmpdir = _create_temp_job_manager()
    
    try:
        job_id = manager.create_job(name="Test", provider="kaggle")
    
        # Update
        updated = manager.update_job(
            job_id=job_id,
            status="running",
            progress=0.5,
        )
    
        assert updated is not None
        assert updated["status"] == "running"
        assert updated["progress"] == 0.5
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_job_manager_list():
    """Test job listing."""
    manager, tmpdir = _create_temp_job_manager()
    
    try:
        # Create multiple jobs
        manager.create_job(name="Job 1", provider="kaggle")
        manager.create_job(name="Job 2", provider="colab")
        manager.create_job(name="Job 3", provider="kaggle")
    
        # List all
        jobs = manager.list_jobs()
        assert len(jobs) == 3
    
        # Filter by provider
        kaggle_jobs = manager.list_jobs(provider="kaggle")
        assert len(kaggle_jobs) == 2
    
        colab_jobs = manager.list_jobs(provider="colab")
        assert len(colab_jobs) == 1
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_workflow_executor():
    """Test WorkflowExecutor basic execution."""
    executor = WorkflowExecutor()
    
    workflow = {
        "plan_id": "test_plan",
        "objective": "Test objective",
        "steps": [
            {"step_id": "step1", "description": "First step", "estimated_duration_minutes": 1},
            {"step_id": "step2", "description": "Second step", "estimated_duration_minutes": 1},
        ],
    }
    
    result = executor.execute(workflow)
    
    assert result["status"] == "completed"
    assert len(result["steps_completed"]) == 2
    assert len(result["steps_failed"]) == 0
    assert "workflow_id" in result
    assert "duration_seconds" in result


def test_workflow_executor_publish_requires_approval():
    """Test that publishing step requires approval."""
    executor = WorkflowExecutor()
    
    workflow = {
        "plan_id": "test_plan",
        "objective": "Test objective",
        "steps": [
            {"step_id": "publish_optional", "description": "Publish model", "estimated_duration_minutes": 1},
        ],
    }
    
    result = executor.execute(workflow)
    
    # Publishing step should not succeed (requires approval)
    publish_result = result["step_results"].get("publish_optional", {})
    assert publish_result.get("requires_approval") is True


def test_job_statistics():
    """Test job statistics."""
    manager, tmpdir = _create_temp_job_manager()
    
    try:
        manager.create_job(name="Job 1", provider="kaggle")
        manager.create_job(name="Job 2", provider="kaggle")
        manager.create_job(name="Job 3", provider="colab")
    
        stats = manager.get_job_statistics()
        assert stats["total"] == 3
        assert stats["by_status"]["pending"] == 3
        assert stats["by_provider"]["kaggle"] == 2
        assert stats["by_provider"]["colab"] == 1
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)