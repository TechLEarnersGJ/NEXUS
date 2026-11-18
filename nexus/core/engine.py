"""
Core engine for NEXUS.

Provides the main engine that orchestrates all components and manages the
complete model development workflow.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from nexus.core.config import Config
from nexus.core.exceptions import JobStateError, ProviderError
from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger
from nexus.core.state import JobManager, JobStatus
from nexus.providers.manager import ProviderManager
from nexus.models.discovery import ModelDiscovery
from nexus.datasets.discovery import DatasetDiscovery
from nexus.training.engine import TrainingEngine
from nexus.evaluation.benchmarks import BenchmarkSuite


class Engine:
    """Main engine for NEXUS model development."""
    
    def __init__(self, config: Optional[Config] = None):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.config = config or Config()
        
        # Configure the global logger
        from nexus.core.logger import logger
        log_level = self.config.get('log_level', 'INFO')
        if hasattr(log_level, 'value'):
            log_level = log_level.value
        elif hasattr(log_level, 'name'):
            log_level = log_level.name
        else:
            log_level = str(log_level)
        
        log_format = self.config.get('log_format', 'json')
        if hasattr(log_format, 'value'):
            log_format = log_format.value
        elif hasattr(log_format, 'name'):
            log_format = log_format.name
        else:
            log_format = str(log_format)
        
        logger.configure(
            log_dir=str(Path(self.config.get('workspace_base', 'workspace')) / 'logs'),
            log_level=log_level,
            log_format=log_format,
            max_file_size_mb=100,  # Simplified part for M0-M2
            backup_count=5
        )
        
        self.logger = logger
        
        # Initialize components
        self.provider_manager = ProviderManager(self.config)
        self.model_discovery = ModelDiscovery()
        self.dataset_discovery = DatasetDiscovery()
        self.training_engine = TrainingEngine(self.config)
        self.benchmark_suite = BenchmarkSuite()
        
        self.nexus_logger.log_event(
            "engine_initialized",
            {
                "config_sources": self.config.get_all_sources(),
                "log_level": self.config.get('log_level'),
                "log_format": self.config.get('log_format'),
                "workspace_base": self.config.get('workspace_base'),
            }
        )
    
    def create_job(self, name: str, description: Optional[str] = None,
                   provider: Optional[str] = None,
                   config: Optional[Dict[str, Any]] = None) -> str:
        """Create a new job."""
        job_manager = JobManager()
        
        # Generate job ID
        job_id = self._generate_job_id(name)
        
        # Create job metadata
        from nexus.core.state import JobMetadata, JobPriority
        metadata = JobMetadata(
            job_id=job_id,
            name=name,
            description=description,
            provider=provider,
            config=config or {}
        )
        
        # Save job
        job_manager.create_job(job_id, metadata)
        
        self.nexus_logger.log_event(
            "job_created",
            {
                "job_id": job_id,
                "name": name,
                "description": description,
                "provider": provider,
                "config": config
            }
        )
        
        return job_id
    
    def get_job(self, job_id: str):
        """Get job by ID."""
        job_manager = JobManager()
        return job_manager.get_job(job_id)
    
    def update_job(self, job_id: str, status: JobStatus,
                   metadata_updates: Optional[Dict[str, Any]] = None,
                   event_type: Optional[str] = None,
                   event_details: Optional[Dict[str, Any]] = None):
        """Update job status."""
        job_manager = JobManager()
        return job_manager.update_job(
            job_id, status, metadata_updates, event_type, event_details
        )
    
    def list_jobs(self, status: Optional[JobStatus] = None,
                  provider: Optional[str] = None):
        """List jobs with optional filtering."""
        job_manager = JobManager()
        return job_manager.list_jobs(status, provider)
    
    def delete_job(self, job_id: str):
        """Delete job."""
        job_manager = JobManager()
        job_manager.delete_job(job_id)
    
    def discover_models(self, query: Optional[str] = None,
                       max_models: int = 10) -> list:
        """Discover models."""
        self.nexus_logger.log_event(
            "model_discovery_started",
            {
                "query": query,
                "max_models": max_models
            }
        )
        
        models = self.model_discovery.search(
            query=query,
            max_models=max_models
        )
        
        self.nexus_logger.log_event(
            "model_discovery_completed",
            {
                "query": query,
                "max_models": max_models,
                "models_found": len(models),
                "models": [
                    {
                        "id": model.id,
                        "model_id": model.model_id,
                        "arch": model.architecture,
                        "params": model.parameter_count,
                        "license": model.license,
                        "tags": model.tags
                    } for model in models
                ]
            }
        )
        
        return models
    
    def discover_datasets(self, query: Optional[str] = None,
                          max_datasets: int = 10) -> list:
        """Discover datasets."""
        self.nexus_logger.log_event(
            "dataset_discovery_started",
            {
                "query": query,
                "max_datasets": max_datasets
            }
        )
        
        datasets = self.dataset_discovery.search(
            query=query,
            max_datasets=max_datasets
        )
        
        self.nexus_logger.log_event(
            "dataset_discovery_completed",
            {
                "query": query,
                "max_datasets": max_datasets,
                "datasets_found": len(datasets),
                "datasets": [
                    {
                        "id": dataset.id,
                        "dataset_id": dataset.dataset_id,
                        "desc": dataset.description,
                        "size": dataset.size,
                        "license": dataset.license,
                        "tags": dataset.tags
                    } for dataset in datasets
                ]
            }
        )
        
        return datasets
    
    def _generate_job_id(self, name: str) -> str:
        """Generate a deterministic job ID from name."""
        import hashlib
        import uuid
        
        # Hash the name for consistency
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        
        # Get timestamp component
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        
        # Combine with UUID for uniqueness
        unique_suffix = str(uuid.uuid4())[:8]
        
        return f"job_{timestamp}_{name_hash}_{unique_suffix}"


# Import here to avoid circular dependencies
from datetime import datetime