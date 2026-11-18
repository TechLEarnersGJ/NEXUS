"""
Orchestration job manager for NEXUS.

Manages the lifecycle of model development jobs, including creation,
status tracking, persistence, and integration with the state manager.
"""

import time
from typing import Any, Dict, List, Optional

from nexus.core.exceptions import JobStateError
from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger
from nexus.core.state import JobManager as StateJobManager, JobStatus, JobMetadata, JobPriority


class OrchestrationJobManager:
    """Manages jobs at the orchestration layer."""
    
    def __init__(
        self,
        state_job_manager: Optional[StateJobManager] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        self.config = config or {}
        self.state_job_manager = state_job_manager or StateJobManager()
        
        # In-memory cache for active jobs
        self._active_jobs: Dict[str, Dict[str, Any]] = {}
    
    def create_job(
        self,
        name: str,
        description: Optional[str] = None,
        provider: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        priority: int = 1,
    ) -> str:
        """Create a new orchestration job."""
        self.nexus_logger.log_event(
            "orchestration_job_create_start",
            {
                "name": name,
                "provider": provider,
                "priority": priority,
            }
        )
        
        # Generate job ID
        import hashlib
        job_id = f"job_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        # Create job metadata
        metadata = JobMetadata(
            job_id=job_id,
            name=name,
            description=description,
            provider=provider,
            config=config or {},
            priority=JobPriority(priority) if 1 <= priority <= 4 else JobPriority.NORMAL,
        )
        
        # Save to state manager
        self.state_job_manager.create_job(job_id, metadata)
        
        # Track in active jobs
        self._active_jobs[job_id] = {
            "job_id": job_id,
            "name": name,
            "description": description,
            "provider": provider,
            "config": config or {},
            "status": "pending",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        
        self.nexus_logger.log_event(
            "orchestration_job_created",
            {
                "job_id": job_id,
                "name": name,
                "provider": provider,
            }
        )
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job details by ID."""
        # Check active jobs first
        if job_id in self._active_jobs:
            return dict(self._active_jobs[job_id])
        
        # Try state manager
        try:
            metadata = self.state_job_manager.get_job(job_id)
            return {
                "job_id": metadata.job_id,
                "name": metadata.name,
                "description": metadata.description,
                "provider": metadata.provider,
                "config": metadata.config,
                "status": metadata.status.value,
                "progress": metadata.progress,
                "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
                "updated_at": metadata.updated_at.isoformat() if metadata.updated_at else None,
                "error": metadata.error,
                "results": metadata.results,
            }
        except JobStateError:
            return None
    
    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        error: Optional[str] = None,
        results: Optional[Dict[str, Any]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update job status and details."""
        self.nexus_logger.log_event(
            "orchestration_job_update_start",
            {
                "job_id": job_id,
                "status": status,
                "progress": progress,
            }
        )
        
        # Update active jobs cache
        if job_id in self._active_jobs:
            job = self._active_jobs[job_id]
            if status is not None:
                job["status"] = status
            if progress is not None:
                job["progress"] = progress
            if error is not None:
                job["error"] = error
            if results is not None:
                job["results"] = results
            if metadata_updates:
                job.update(metadata_updates)
            job["updated_at"] = time.time()
        
        # Update state manager
        try:
            metadata = self.state_job_manager.get_job(job_id)
            
            # Convert status string to enum
            status_enum = None
            if status:
                try:
                    status_enum = JobStatus(status)
                except ValueError:
                    pass
            
            update_kwargs = {}
            if status_enum:
                update_kwargs["status"] = status_enum
            if progress is not None:
                update_kwargs["progress"] = progress
            if error is not None:
                update_kwargs["error"] = error
            if results is not None:
                update_kwargs["results"] = results
            if metadata_updates:
                for k, v in metadata_updates.items():
                    if hasattr(metadata, k):
                        setattr(metadata, k, v)
            
            if update_kwargs or metadata_updates:
                self.state_job_manager.update_job(
                    job_id,
                    status=status_enum,
                    metadata_updates=metadata_updates or {},
                )
            
        except JobStateError:
            pass
        
        return self.get_job(job_id)
    
    def list_jobs(
        self,
        status: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all jobs with optional filtering."""
        # Get all jobs from state manager
        state_jobs = self.state_job_manager.list_jobs()
        
        results = []
        for job in state_jobs:
            # Filter by status
            if status and job.status.value != status:
                continue
            
            # Filter by provider
            if provider and job.provider != provider:
                continue
            
            results.append({
                "job_id": job.job_id,
                "name": job.name,
                "description": job.description,
                "provider": job.provider,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.created_at.isoformat() if job.created_at else None,
            })
        
        return results
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        self.nexus_logger.log_event(
            "orchestration_job_delete_start",
            {
                "job_id": job_id,
            }
        )
        
        # Remove from active jobs
        if job_id in self._active_jobs:
            del self._active_jobs[job_id]
        
        # Delete from state manager
        try:
            self.state_job_manager.delete_job(job_id)
        except JobStateError:
            return False
        
        self.nexus_logger.log_event(
            "orchestration_job_deleted",
            {
                "job_id": job_id,
            }
        )
        
        return True
    
    def get_job_events(self, job_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get job event history."""
        try:
            events = self.state_job_manager.get_job_events(job_id, limit=limit)
            return [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "details": e.details,
                    "user_id": e.user_id,
                }
                for e in events
            ]
        except JobStateError:
            return []
    
    def get_active_job_count(self) -> int:
        """Get the count of active (non-terminal) jobs."""
        active_statuses = {"pending", "queued", "running", "submitted"}
        count = 0
        for job in self.list_jobs():
            if job.get("status") in active_statuses:
                count += 1
        return count
    
    def get_job_statistics(self) -> Dict[str, Any]:
        """Get overall job statistics."""
        jobs = self.list_jobs()
        
        stats = {
            "total": len(jobs),
            "by_status": {},
            "by_provider": {},
        }
        
        for job in jobs:
            status = job.get("status", "unknown")
            provider = job.get("provider", "unknown")
            
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            stats["by_provider"][provider] = stats["by_provider"].get(provider, 0) + 1
        
        return stats