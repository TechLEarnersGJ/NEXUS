"""
Job state management for NEXUS.

Provides persistent job state storage, management, and validation
for tracking the progress of model development jobs.
"""

import json
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import hashlib

from nexus.core.exceptions import JobStateError


class NexusJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for NEXUS objects."""
    
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value if hasattr(obj, 'value') else str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class JobStatus(Enum):
    """Job status enumeration."""
    
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    UNKNOWN = "unknown"


class JobPriority(Enum):
    """Job priority enumeration."""
    
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class JobMetadata:
    """Job metadata."""
    
    job_id: str
    name: str
    description: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Provider information
    provider: Optional[str] = None
    provider_job_id: Optional[str] = None
    
    # Job configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Job progress
    progress: float = 0.0
    current_step: Optional[str] = None
    steps_total: Optional[int] = None
    steps_completed: Optional[int] = None
    
    # Results
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    # File paths
    workspace_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    output_path: Optional[str] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    parent_job_id: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class JobEvent:
    """Job event for state changes."""
    
    event_id: str
    job_id: str
    event_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None


class JobManager:
    """Manager for persistent job state."""
    
    def __init__(self, storage_dir: str = "workspace/jobs"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Index files for quick lookup
        self._index_file = self.storage_dir / "index.json"
        self._jobs_index: Dict[str, Dict[str, Any]] = {}
        
        # Load index
        self._load_index()
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def _load_index(self):
        """Load the jobs index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r') as f:
                    self._jobs_index = json.load(f)
            except Exception:
                self._jobs_index = {}
    
    def _save_index(self):
        """Save the jobs index to disk."""
        try:
            with open(self._index_file, 'w') as f:
                json.dump(self._jobs_index, f, indent=2, cls=NexusJSONEncoder)
        except Exception as e:
            raise JobStateError(f"Failed to save jobs index: {str(e)}")
    
    def _get_job_file_path(self, job_id: str) -> Path:
        """Get the file path for a job."""
        # Sanitize job_id for filesystem use
        safe_job_id = ''.join(c for c in job_id if c.isalnum() or c in '-_')
        return self.storage_dir / f"{safe_job_id}.json"
    
    def create_job(self, job_id: str, metadata: JobMetadata) -> JobMetadata:
        """Create a new job."""
        with self._lock:
            if not job_id:
                raise JobStateError("Job ID cannot be empty")
            
            if self.exists(job_id):
                raise JobStateError(f"Job already exists: {job_id}")
            
            # Generate deterministic event ID
            event_id = self._generate_event_id(job_id, "create")
            
            # Create job event
            event = JobEvent(
                event_id=event_id,
                job_id=job_id,
                event_type="job_created",
                details={"metadata": metadata.__dict__}
            )
            
            # Save metadata
            metadata.created_at = datetime.utcnow()
            self._save_job_metadata(job_id, metadata)
            
            # Save event
            self._save_job_event(job_id, event)
            
            # Update index
            self._jobs_index[job_id] = {
                "name": metadata.name,
                "status": metadata.status.value,
                "created_at": metadata.created_at.isoformat(),
                "provider": metadata.provider,
                "priority": metadata.priority.value
            }
            
            self._save_index()
            
            return metadata
    
    def get_job(self, job_id: str) -> JobMetadata:
        """Get job metadata."""
        with self._lock:
            if not self.exists(job_id):
                raise JobStateError(f"Job not found: {job_id}")
            
            return self._load_job_metadata(job_id)
    
    def update_job(self, job_id: str, status: Optional[JobStatus] = None,
                   metadata_updates: Optional[Dict[str, Any]] = None,
                   event_type: Optional[str] = None,
                   event_details: Optional[Dict[str, Any]] = None) -> JobMetadata:
        """Update job metadata and create an event."""
        with self._lock:
            if not self.exists(job_id):
                raise JobStateError(f"Job not found: {job_id}")
            
            # Load current metadata
            job_metadata = self._load_job_metadata(job_id)
            
            # Update status if provided
            if status is not None:
                if status not in JobStatus:
                    raise JobStateError(f"Invalid job status: {status}")
                
                old_status = job_metadata.status
                job_metadata.status = status
                job_metadata.updated_at = datetime.utcnow()
                
                if status == JobStatus.RUNNING and old_status == JobStatus.QUEUED:
                    job_metadata.started_at = datetime.utcnow()
                
                if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                    job_metadata.completed_at = datetime.utcnow()
            
            # Apply other metadata updates
            if metadata_updates:
                for key, value in metadata_updates.items():
                    if hasattr(job_metadata, key):
                        setattr(job_metadata, key, value)
            
            # Save updated metadata
            self._save_job_metadata(job_id, job_metadata)
            
            # Create event if requested
            if event_type:
                event_id = self._generate_event_id(job_id, event_type)
                event = JobEvent(
                    event_id=event_id,
                    job_id=job_id,
                    event_type=event_type,
                    details=event_details or {}
                )
                self._save_job_event(job_id, event)
            
            # Update index
            self._jobs_index[job_id].update({
                "status": job_metadata.status.value,
                "updated_at": job_metadata.updated_at.isoformat() if job_metadata.updated_at else None,
                "progress": job_metadata.progress
            })
            
            self._save_index()
            
            return job_metadata
    
    def delete_job(self, job_id: str):
        """Delete a job."""
        with self._lock:
            if not self.exists(job_id):
                raise JobStateError(f"Job not found: {job_id}")
            
            # Remove from index
            del self._jobs_index[job_id]
            self._save_index()
            
            # Delete job file
            job_file = self._get_job_file_path(job_id)
            if job_file.exists():
                job_file.unlink()
    
    def exists(self, job_id: str) -> bool:
        """Check if job exists."""
        return job_id in self._jobs_index
    
    def list_jobs(self, status: Optional[JobStatus] = None, 
                  provider: Optional[str] = None) -> List[JobMetadata]:
        """List jobs with optional filtering."""
        with self._lock:
            jobs = []
            
            for job_id in self._jobs_index.keys():
                try:
                    job = self._load_job_metadata(job_id)
                    
                    # Filter by status
                    if status is not None and job.status != status:
                        continue
                    
                    # Filter by provider
                    if provider is not None and job.provider != provider:
                        continue
                    
                    jobs.append(job)
                    
                except JobStateError:
                    # Skip jobs that can't be loaded
                    continue
            
            return jobs
    
    def _load_job_metadata(self, job_id: str) -> JobMetadata:
        """Load job metadata from disk."""
        job_file = self._get_job_file_path(job_id)
        
        if not job_file.exists():
            raise JobStateError(f"Job metadata not found: {job_id}")
        
        try:
            with open(job_file, 'r') as f:
                data = json.load(f)
            
            # Convert string dates back to datetime
            if 'created_at' in data and isinstance(data['created_at'], str):
                data['created_at'] = datetime.fromisoformat(data['created_at'])
            
            if 'updated_at' in data and isinstance(data['updated_at'], str):
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])
            
            if 'started_at' in data and isinstance(data['started_at'], str):
                data['started_at'] = datetime.fromisoformat(data['started_at'])
            
            if 'completed_at' in data and isinstance(data['completed_at'], str):
                data['completed_at'] = datetime.fromisoformat(data['completed_at'])
            
            # Convert status string to enum
            if 'status' in data and isinstance(data['status'], str):
                data['status'] = JobStatus(data['status'])
            
            # Convert priority string to enum
            if 'priority' in data and isinstance(data['priority'], int):
                data['priority'] = JobPriority(data['priority'])
            
            return JobMetadata(**data)
            
        except Exception as e:
            raise JobStateError(f"Failed to load job metadata for {job_id}: {str(e)}")
    
    def _save_job_metadata(self, job_id: str, metadata: JobMetadata):
        """Save job metadata to disk."""
        job_file = self._get_job_file_path(job_id)
        
        try:
            # Convert datetime objects to strings
            metadata_dict = metadata.__dict__.copy()
            
            for key, value in metadata_dict.items():
                if isinstance(value, datetime):
                    metadata_dict[key] = value.isoformat()
                elif isinstance(value, JobStatus):
                    metadata_dict[key] = value.value
                elif isinstance(value, JobPriority):
                    metadata_dict[key] = value.value
            
            with open(job_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2, cls=NexusJSONEncoder)
                
        except Exception as e:
            raise JobStateError(f"Failed to save job metadata for {job_id}: {str(e)}")
    
    def _load_job_event(self, job_id: str, event_id: str) -> JobEvent:
        """Load job event from disk."""
        events_file = self._get_events_file_path(job_id)
        
        if not events_file.exists():
            raise JobStateError(f"Job events not found: {job_id}")
        
        try:
            with open(events_file, 'r') as f:
                events = json.load(f)
            
            for event_data in events:
                if event_data.get('event_id') == event_id:
                    if 'timestamp' in event_data and isinstance(event_data['timestamp'], str):
                        event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                    
                    return JobEvent(**event_data)
            
            raise JobStateError(f"Event not found: {event_id} for job {job_id}")
            
        except Exception as e:
            raise JobStateError(f"Failed to load job event {event_id}: {str(e)}")
    
    def _save_job_event(self, job_id: str, event: JobEvent):
        """Save job event to disk."""
        events_file = self._get_events_file_path(job_id)
        
        try:
            # Load existing events if file exists
            if events_file.exists():
                with open(events_file, 'r') as f:
                    events = json.load(f)
            else:
                events = []
            
            # Add new event
            event_data = event.__dict__.copy()
            if isinstance(event_data.get('timestamp'), datetime):
                event_data['timestamp'] = event_data['timestamp'].isoformat()
            
            events.append(event_data)
            
            # Write back to file
            with open(events_file, 'w') as f:
                json.dump(events, f, indent=2, cls=NexusJSONEncoder)
                
        except Exception as e:
            raise JobStateError(f"Failed to save job event {event.event_id}: {str(e)}")
    
    def _get_events_file_path(self, job_id: str) -> Path:
        """Get the file path for job events."""
        safe_job_id = ''.join(c for c in job_id if c.isalnum() or c in '-_')
        return self.storage_dir / f"{safe_job_id}_events.json"
    
    def _generate_event_id(self, job_id: str, event_type: str) -> str:
        """Generate a deterministic event ID."""
        data = f"{job_id}:{event_type}:{datetime.utcnow().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def get_job_events(self, job_id: str, limit: Optional[int] = None) -> List[JobEvent]:
        """Get job events."""
        with self._lock:
            if not self.exists(job_id):
                raise JobStateError(f"Job not found: {job_id}")
            
            events_file = self._get_events_file_path(job_id)
            
            if not events_file.exists():
                return []
            
            try:
                with open(events_file, 'r') as f:
                    events_data = json.load(f)
                
                # Convert to JobEvent objects
                events = []
                for event_data in events_data:
                    if 'timestamp' in event_data and isinstance(event_data['timestamp'], str):
                        event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                    
                    events.append(JobEvent(**event_data))
                
                # Sort by timestamp
                events.sort(key=lambda e: e.timestamp)
                
                # Apply limit
                if limit is not None:
                    events = events[-limit:]
                
                return events
                
            except Exception as e:
                raise JobStateError(f"Failed to load job events for {job_id}: {str(e)}")


class JobStateValidator:
    """Validator for job state transitions."""
    
    # Define valid state transitions
    VALID_TRANSITIONS = {
        JobStatus.PENDING: [JobStatus.QUEUED, JobStatus.CANCELLED],
        JobStatus.QUEUED: [JobStatus.RUNNING, JobStatus.PENDING, JobStatus.CANCELLED],
        JobStatus.RUNNING: [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.PAUSED],
        JobStatus.PAUSED: [JobStatus.RUNNING, JobStatus.CANCELLED],
        JobStatus.COMPLETED: [],
        JobStatus.FAILED: [],
        JobStatus.CANCELLED: [],
        JobStatus.UNKNOWN: [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING]
    }
    
    @staticmethod
    def is_valid_transition(from_status: JobStatus, to_status: JobStatus) -> bool:
        """Check if a state transition is valid."""
        return to_status in JobStateValidator.VALID_TRANSITIONS.get(from_status, [])
    
    @staticmethod
    def get_allowed_transitions(status: JobStatus) -> List[JobStatus]:
        """Get allowed transitions for a given status."""
        return JobStateValidator.VALID_TRANSITIONS.get(status, [])