"""
Command-line interface for NEXUS.

Provides the main CLI entry point and commands for interacting with the
NEXUS autonomous AI model development platform.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click

from nexus.core.config import Config
from nexus.core.engine import Engine
from nexus.core.exceptions import ConfigError, AuthError
from nexus.core.logger import get_logger


class NexusCLI:
    """Command-line interface for NEXUS."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.config = Config()
        
        # Load config files
        config_dir = Path("config")
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                try:
                    self.config.load_from_file(config_file, config_file.stem)
                except Exception as e:
                    self.logger.warning(f"Failed to load config from {config_file}: {str(e)}")
        
        self.engine = Engine(self.config)
    
    def status(self):
        """Display NEXUS system status."""
        click.echo("NEXUS Status")
        click.echo("=" * 50)
        
        # Check configuration
        config_sources = self.engine.config.get_all_sources()
        click.echo(f"Configuration sources: {len(config_sources)}")
        for source in config_sources:
            click.echo(f"  - {source}")
        
        # Check providers
        providers = self.engine.config.get("providers", {})
        click.echo(f"\nConfigured providers: {len(providers)}")
        for provider_name, provider_config in providers.items():
            if provider_config.get("enabled", False):
                status = "Enabled"
                click.echo(f"  - {provider_name}: {status}")
        
        # Check workspace
        workspace_base = self.engine.config.get("workspace_base", "workspace")
        workspace_path = Path(workspace_base)
        click.echo(f"\nWorkspace: {workspace_path.absolute()}")
        click.echo(f"  Exists: {workspace_path.exists()}")
        
        # Show configuration validation
        config_errors = self.engine.config.validate()
        if config_errors:
            click.echo(f"\nConfiguration errors: {len(config_errors)}")
            for error in config_errors:
                click.echo(f"  - {error}")
        else:
            click.echo("\nConfiguration: Valid")
    
    def show_config(self):
        """Display current configuration."""
        click.echo("NEXUS Configuration")
        click.echo("=" * 50)
        
        # Show effective configuration
        config_dict = self.config.to_dict()
        click.echo("Effective configuration:")
        
        import yaml
        click.echo(yaml.dump(config_dict, default_flow_style=False))
    
    def providers(self):
        """Display provider information."""
        click.echo("NEXUS Providers")
        click.echo("=" * 50)
        
        providers = self.engine.config.get("providers", {})
        
        for provider_name, provider_config in providers.items():
            if not provider_config.get("enabled", False):
                continue
            
            click.echo(f"\n{provider_name.upper()}:")
            click.echo(f"  Description: {provider_config.get('description', 'No description')}")
            click.echo(f"  Priority: {provider_config.get('priority', 1)}")
            click.echo(f"  Timeout: {provider_config.get('timeout_seconds', 30)} seconds")
    
    def jobs(self):
        """List all jobs."""
        from nexus.core.state import JobManager, JobStatus
        
        job_manager = JobManager()
        
        click.echo("NEXUS Jobs")
        click.echo("=" * 50)
        
        jobs = job_manager.list_jobs()
        
        if not jobs:
            click.echo("No jobs found")
            return
        
        click.echo(f"\nTotal jobs: {len(jobs)}")
        
        # Group jobs by status
        jobs_by_status = {}
        for job in jobs:
            status = job.status
            if status not in jobs_by_status:
                jobs_by_status[status] = []
            jobs_by_status[status].append(job)
        
        # Display jobs by status
        for status, status_jobs in jobs_by_status.items():
            click.echo(f"\n{status.value.upper()} ({len(status_jobs)}):")
            
            for job in status_jobs[:10]:  # Show first 10 jobs per status
                click.echo(f"  - {job.name} (ID: {job.job_id})")
                click.echo(f"    Provider: {job.provider or 'N/A'}")
                click.echo(f"    Progress: {job.progress * 100:.1f}%")
    
    def create(self, name: str, description: Optional[str] = None,
               provider: Optional[str] = None, config: Optional[str] = None):
        """Create a new job."""
        try:
            # Parse config if provided
            job_config = {}
            if config:
                import yaml
                with open(config, 'r') as f:
                    job_config = yaml.safe_load(f)
            
            # Create job
            job_id = self.engine.create_job(
                name=name,
                description=description,
                provider=provider,
                config=job_config
            )
            
            click.echo(f"Job created successfully!")
            click.echo(f"Job ID: {job_id}")
            click.echo(f"Name: {name}")
            if description:
                click.echo(f"Description: {description}")
            if provider:
                click.echo(f"Provider: {provider}")
                
        except Exception as e:
            click.echo(f"Error creating job: {str(e)}", err=True)
            sys.exit(1)
    
    def plan(self, objective: str):
        """Plan a workflow for the given objective."""
        try:
            from nexus.orchestration.planner import WorkflowPlanner
            
            planner = WorkflowPlanner(
                model_discovery=self.engine.model_discovery,
                dataset_discovery=self.engine.dataset_discovery,
            )
            
            # Build model requirements
            model_requirements = {
                "objective": objective,
                "max_parameters": "1B",
                "languages": ["python", "javascript"],
                "capabilities": ["conversational", "code_generation"],
                "training_method": "qlora",
                "preferred_providers": ["kaggle", "colab"],
                "model_registry": "huggingface",
                "budget": "free",
            }
            
            dataset_requirements = {
                "objective": objective,
                "types": ["instruction", "code", "conversational"],
                "languages": ["python", "javascript"],
            }
            
            workflow = planner.plan(
                objective=objective,
                model_requirements=model_requirements,
                dataset_requirements=dataset_requirements,
            )
            
            click.echo("Workflow planned successfully!")
            click.echo(f"Objective: {objective}")
            click.echo(f"Plan ID: {workflow.get('plan_id', 'N/A')}")
            click.echo(f"Estimated duration: {workflow.get('estimated_duration_minutes', 'N/A')} minutes")
            click.echo(f"Estimated GPU hours: {workflow.get('estimated_gpu_hours', 'N/A')}")
            click.echo(f"Provider sequence: {', '.join(workflow.get('provider_sequence', []))}")
            click.echo(f"Steps: {len(workflow.get('steps', []))}")
            
            # Show steps
            click.echo("\nExecution steps:")
            for i, step in enumerate(workflow.get("steps", []), 1):
                click.echo(f"  {i}. {step.get('description', step.get('step_id', 'unknown'))}")
                
        except Exception as e:
            click.echo(f"Error planning workflow: {str(e)}", err=True)
            sys.exit(1)
    
    def train(self, job_id: str):
        """Train a model for the given job."""
        try:
            # Get job
            job = self.engine.get_job(job_id)
            if not job:
                click.echo(f"Job {job_id} not found", err=True)
                sys.exit(1)
            
            click.echo(f"Training job: {job_id}")
            click.echo(f"Name: {job.name}")
            click.echo(f"Provider: {job.provider or 'N/A'}")
            
            # Check if job has a plan
            config = job.config or {}
            plan = config.get("plan", {})
            
            if not plan:
                click.echo("No training plan found for this job. Run 'nexus plan' first.")
                sys.exit(1)
            
            click.echo(f"Plan: {plan.get('plan_id', 'N/A')}")
            click.echo(f"Steps: {len(plan.get('steps', []))}")
            
            # In a full implementation, this would submit the training job
            # to the selected provider and monitor execution
            click.echo("\nTraining would execute the following steps:")
            for i, step in enumerate(plan.get("steps", []), 1):
                click.echo(f"  {i}. {step.get('description', step.get('step_id', 'unknown'))}")
            
            click.echo("\nNote: Full training execution requires cloud provider credentials.")
            click.echo("Use 'nexus providers' to check provider availability.")
            
        except Exception as e:
            click.echo(f"Error training job: {str(e)}", err=True)
            sys.exit(1)
    
    def evaluate(self, job_id: str):
        """Evaluate a trained model."""
        try:
            # Get job
            job = self.engine.get_job(job_id)
            if not job:
                click.echo(f"Job {job_id} not found", err=True)
                sys.exit(1)
            
            click.echo(f"Evaluating job: {job_id}")
            click.echo(f"Name: {job.name}")
            
            # In a full implementation, this would run the evaluation benchmarks
            click.echo("\nEvaluation would run the following benchmarks:")
            click.echo("  1. Code generation (Python, JavaScript)")
            click.echo("  2. Chat quality")
            click.echo("  3. Safety evaluation")
            click.echo("  4. Reasoning tests")
            click.echo("  5. Regression comparison (baseline vs fine-tuned)")
            
            click.echo("\nNote: Full evaluation requires a trained model checkpoint.")
            
        except Exception as e:
            click.echo(f"Error evaluating job: {str(e)}", err=True)
            sys.exit(1)
    
    def export(self, job_id: str):
        """Export a trained model."""
        try:
            # Get job
            job = self.engine.get_job(job_id)
            if not job:
                click.echo(f"Job {job_id} not found", err=True)
                sys.exit(1)
            
            click.echo(f"Exporting job: {job_id}")
            click.echo(f"Name: {job.name}")
            
            # In a full implementation, this would export the model
            click.echo("\nExport would include:")
            click.echo("  - Model adapter (LoRA/QLoRA weights)")
            click.echo("  - Tokenizer")
            click.echo("  - Training metadata")
            click.echo("  - Evaluation results")
            click.echo("  - Training report")
            
            click.echo("\nNote: Full export requires a completed training job.")
            
        except Exception as e:
            click.echo(f"Error exporting job: {str(e)}", err=True)
            sys.exit(1)
    
    def publish(self, job_id: str):
        """Publish a trained model to Hugging Face."""
        try:
            # Get job
            job = self.engine.get_job(job_id)
            if not job:
                click.echo(f"Job {job_id} not found", err=True)
                sys.exit(1)
            
            click.echo(f"Publishing job: {job_id}")
            click.echo(f"Name: {job.name}")
            click.echo("\nWARNING: This will publish the model to Hugging Face Hub.")
            click.echo("Make sure you have permission to publish and the model license allows it.")
            
            # Check approval gate
            from nexus.orchestration.approval import ApprovalGate
            gate = ApprovalGate()
            
            result = gate.check_operation_allowed(
                "publish_model",
                auto_create_request=True,
                model_id=job_id,
                job_name=job.name,
            )
            
            if not result["allowed"]:
                click.echo(f"\nApproval required: {result['message']}")
                click.echo(f"Request ID: {result.get('request_id', 'N/A')}")
                click.echo("\nTo approve, run:")
                click.echo(f"  nexus approve {result.get('request_id', 'N/A')} <your_username>")
                sys.exit(1)
            
            click.echo("\nApproval granted. Publishing would include:")
            click.echo("  - Model upload to Hugging Face Hub")
            click.echo("  - Model card generation")
            click.echo("  - License validation")
            click.echo("  - README creation")
            
        except Exception as e:
            click.echo(f"Error publishing job: {str(e)}", err=True)
            sys.exit(1)
    
    def approve(self, request_id: str, approver: str):
        """Approve a pending approval request."""
        try:
            from nexus.orchestration.approval import ApprovalGate
            gate = ApprovalGate()
            
            success = gate.approve_request(request_id, approver)
            
            if success:
                click.echo(f"Approval granted for request: {request_id}")
                click.echo(f"Approver: {approver}")
            else:
                click.echo(f"Failed to approve request: {request_id}", err=True)
                sys.exit(1)
                
        except Exception as e:
            click.echo(f"Error approving request: {str(e)}", err=True)
            sys.exit(1)
    
    def logs(self, job_id: str):
        """Display job logs."""
        try:
            from nexus.core.state import JobManager
            job_manager = JobManager()
            
            # Get job events
            events = job_manager.get_job_events(job_id)
            
            click.echo(f"Job logs: {job_id}")
            click.echo("=" * 50)
            
            if not events:
                click.echo("No events found")
                return
            
            for event in events:
                timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M:%S") if event.timestamp else "N/A"
                click.echo(f"[{timestamp}] {event.event_type}")
                if event.details:
                    for key, value in event.details.items():
                        click.echo(f"    {key}: {value}")
                        
        except Exception as e:
            click.echo(f"Error getting logs: {str(e)}", err=True)
            sys.exit(1)
    
    def quota(self):
        """Display provider quota information."""
        click.echo("NEXUS Provider Quotas")
        click.echo("=" * 50)
        
        providers = self.engine.config.get("providers", {})
        
        for provider_name, provider_config in providers.items():
            if not provider_config.get("enabled", False):
                continue
            
            click.echo(f"\n{provider_name.upper()}:")
            
            # Try to get quota from provider
            try:
                provider = self.engine.provider_manager.get_provider(provider_name)
                if provider:
                    quota = provider.quota()
                    if quota:
                        click.echo(f"  Type: {quota.quota_type}")
                        click.echo(f"  Used: {quota.used}")
                        click.echo(f"  Limit: {quota.limit}")
                        click.echo(f"  Remaining: {quota.remaining}")
                        if quota.refresh_at:
                            import datetime
                            refresh = datetime.datetime.fromtimestamp(quota.refresh_at)
                            click.echo(f"  Refresh: {refresh.strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        click.echo("  Quota information not available")
                else:
                    click.echo("  Provider not initialized")
            except Exception as e:
                click.echo(f"  Error getting quota: {str(e)}")
    
    def approve_requests(self):
        """List pending approval requests."""
        from nexus.orchestration.approval import ApprovalGate
        gate = ApprovalGate()
        
        requests = gate.get_pending_requests()
        
        click.echo("Pending Approval Requests")
        click.echo("=" * 50)
        
        if not requests:
            click.echo("No pending approval requests")
            return
        
        for req in requests:
            click.echo(f"\nRequest ID: {req.request_id}")
            click.echo(f"  Operation: {req.operation_type}")
            click.echo(f"  Description: {req.description}")
            click.echo(f"  Requested: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(req.requested_at))}")
            if req.expires_at:
                click.echo(f"  Expires: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(req.expires_at))}")
            click.echo(f"  Status: {req.status}")


def main():
    """Main CLI entry point."""
    cli_instance = NexusCLI()
    
    @click.group()
    @click.version_option("0.1.0")
    def cli_group():
        """NEXUS - Autonomous AI Model Development Platform"""
        pass
    
    @cli_group.command()
    def doctor():
        """Check system status and configuration."""
        cli_instance.status()
    
    @cli_group.command()
    def status():
        """Display NEXUS system status."""
        cli_instance.status()
    
    @cli_group.command(name="config")
    def show_config():
        """Display current configuration."""
        cli_instance.show_config()
    
    @cli_group.command()
    def providers():
        """Display provider information."""
        cli_instance.providers()
    
    @cli_group.command()
    def jobs():
        """List all jobs."""
        cli_instance.jobs()
    
    @cli_group.command()
    @click.argument("name")
    @click.option("--description", "-d", help="Job description")
    @click.option("--provider", "-p", help="Provider to use")
    @click.option("--config", "-c", help="Configuration file")
    def create(name, description, provider, config):
        """Create a new job."""
        cli_instance.create(name, description, provider, config)
    
    @cli_group.command()
    @click.argument("objective")
    def plan(objective):
        """Plan a workflow for the given objective."""
        cli_instance.plan(objective)
    
    @cli_group.command()
    @click.argument("job_id")
    def train(job_id):
        """Train a model for the given job."""
        cli_instance.train(job_id)
    
    @cli_group.command()
    @click.argument("job_id")
    def evaluate(job_id):
        """Evaluate a trained model."""
        cli_instance.evaluate(job_id)
    
    @cli_group.command()
    @click.argument("job_id")
    def export(job_id):
        """Export a trained model."""
        cli_instance.export(job_id)
    
    @cli_group.command()
    @click.argument("job_id")
    def publish(job_id):
        """Publish a trained model to Hugging Face."""
        cli_instance.publish(job_id)
    
    @cli_group.command()
    @click.argument("request_id")
    @click.argument("approver")
    def approve(request_id, approver):
        """Approve a pending approval request."""
        cli_instance.approve(request_id, approver)
    
    @cli_group.command()
    def approvals():
        """List pending approval requests."""
        cli_instance.approve_requests()
    
    @cli_group.command()
    @click.argument("job_id")
    def logs(job_id):
        """Display job logs and events."""
        cli_instance.logs(job_id)
    
    @cli_group.command()
    def quota():
        """Display provider quota information."""
        cli_instance.quota()
    
    try:
        cli_group()
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)