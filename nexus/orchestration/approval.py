"""
Approval gate system for NEXUS autonomous orchestration.

Manages user approval requirements for sensitive operations including
model publishing, private dataset access, destructive operations,
and paid resource usage.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus.core.exceptions import ValidationError
from nexus.core.logger import get_logger
from nexus.core.logger import logger as nexus_logger


@dataclass
class ApprovalRequest:
    """Represents a pending approval request."""
    
    request_id: str
    operation_type: str
    description: str
    details: Dict[str, Any]
    status: str = "pending"  # pending, approved, rejected, expired
    requested_at: float = field(default_factory=time.time)
    approved_at: Optional[float] = None
    rejected_at: Optional[float] = None
    approver: Optional[str] = None
    reason: Optional[str] = None
    expires_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if the approval request has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def approve(self, approver: str) -> None:
        """Approve the request."""
        self.status = "approved"
        self.approver = approver
        self.approved_at = time.time()
    
    def reject(self, approver: str, reason: str) -> None:
        """Reject the request."""
        self.status = "rejected"
        self.approver = approver
        self.reason = reason
        self.rejected_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "operation_type": self.operation_type,
            "description": self.description,
            "details": self.details,
            "status": self.status,
            "requested_at": self.requested_at,
            "approved_at": self.approved_at,
            "rejected_at": self.rejected_at,
            "approver": self.approver,
            "reason": self.reason,
            "expires_at": self.expires_at,
            "is_expired": self.is_expired(),
        }


class ApprovalGate:
    """Manages approval gates for sensitive operations."""
    
    # Operation types that require explicit approval
    APPROVAL_REQUIRED_OPERATIONS = {
        "publish_model": "Publish a model to Hugging Face Hub",
        "use_private_dataset": "Access or use a private dataset",
        "destructive_operation": "Perform a destructive operation (delete model, dataset, etc.)",
        "paid_resource": "Access paid cloud resources or services",
        "account_settings_change": "Change account settings or credentials",
    }
    
    def __init__(
        self,
        require_approval_for: Optional[List[str]] = None,
        auto_approve: bool = False,
        default_ttl_hours: int = 24,
    ):
        self.logger = get_logger(__name__)
        self.nexus_logger = nexus_logger
        
        # Operations requiring approval
        self.require_approval_for = set(require_approval_for or list(self.APPROVAL_REQUIRED_OPERATIONS.keys()))
        
        # Auto-approve mode (for testing only)
        self.auto_approve = auto_approve
        
        # Default TTL for approval requests
        self.default_ttl_hours = default_ttl_hours
        
        # Pending approval requests
        self._pending_requests: Dict[str, ApprovalRequest] = {}
        
        # Approval history
        self._approval_history: List[ApprovalRequest] = []
    
    def requires_approval(self, operation_type: str) -> bool:
        """Check if an operation type requires explicit approval."""
        return operation_type in self.require_approval_for
    
    def request_approval(
        self,
        operation_type: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        ttl_hours: Optional[int] = None,
    ) -> ApprovalRequest:
        """Create an approval request for a sensitive operation."""
        import hashlib
        
        request_id = f"approval_{int(time.time())}_{hashlib.md5(description.encode()).hexdigest()[:8]}"
        
        ttl = ttl_hours or self.default_ttl_hours
        expires_at = time.time() + (ttl * 3600)
        
        request = ApprovalRequest(
            request_id=request_id,
            operation_type=operation_type,
            description=description,
            details=details or {},
            expires_at=expires_at,
        )
        
        # Auto-approve if enabled
        if self.auto_approve:
            request.approve("system")
        
        # Store the request
        self._pending_requests[request_id] = request
        
        self.nexus_logger.log_event(
            "approval_requested",
            {
                "request_id": request_id,
                "operation_type": operation_type,
                "description": description,
                "expires_at": expires_at,
            }
        )
        
        return request
    
    def approve_request(
        self,
        request_id: str,
        approver: str,
    ) -> bool:
        """Approve a pending request."""
        if request_id not in self._pending_requests:
            return False
        
        request = self._pending_requests[request_id]
        
        if request.status != "pending":
            return False
        
        if request.is_expired():
            request.status = "expired"
            return False
        
        request.approve(approver)
        
        # Move to history
        self._approval_history.append(request)
        del self._pending_requests[request_id]
        
        self.nexus_logger.log_event(
            "approval_granted",
            {
                "request_id": request_id,
                "operation_type": request.operation_type,
                "approver": approver,
            }
        )
        
        return True
    
    def reject_request(
        self,
        request_id: str,
        approver: str,
        reason: str,
    ) -> bool:
        """Reject a pending request."""
        if request_id not in self._pending_requests:
            return False
        
        request = self._pending_requests[request_id]
        
        if request.status != "pending":
            return False
        
        request.reject(approver, reason)
        
        # Move to history
        self._approval_history.append(request)
        del self._pending_requests[request_id]
        
        self.nexus_logger.log_event(
            "approval_rejected",
            {
                "request_id": request_id,
                "operation_type": request.operation_type,
                "approver": approver,
                "reason": reason,
            }
        )
        
        return True
    
    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get a pending approval request by ID."""
        if request_id in self._pending_requests:
            return self._pending_requests[request_id]
        
        # Check history
        for request in self._approval_history:
            if request.request_id == request_id:
                return request
        
        return None
    
    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        # Clean up expired requests
        expired = [
            rid for rid, req in self._pending_requests.items()
            if req.is_expired()
        ]
        for rid in expired:
            self._pending_requests[rid].status = "expired"
            self._approval_history.append(self._pending_requests[rid])
            del self._pending_requests[rid]
        
        return list(self._pending_requests.values())
    
    def check_operation_allowed(
        self,
        operation_type: str,
        auto_create_request: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Check if an operation is allowed without approval.
        
        Returns a dict with:
            - allowed: bool (True if no approval needed or already approved)
            - requires_approval: bool
            - request_id: str (if approval was created)
            - message: str
        """
        if not self.requires_approval(operation_type):
            return {
                "allowed": True,
                "requires_approval": False,
                "request_id": None,
                "message": "Operation does not require approval",
            }
        
        # Check if there's already an approved request for this operation
        for request in self._approval_history:
            if (request.operation_type == operation_type
                    and request.status == "approved"
                    and not request.is_expired()):
                return {
                    "allowed": True,
                    "requires_approval": True,
                    "request_id": request.request_id,
                    "message": "Operation previously approved",
                }
        
        # Check pending requests
        for request in self._pending_requests.values():
            if (request.operation_type == operation_type
                    and request.status == "pending"
                    and not request.is_expired()):
                return {
                    "allowed": False,
                    "requires_approval": True,
                    "request_id": request.request_id,
                    "message": "Approval request pending",
                }
        
        # Create new approval request if requested
        if auto_create_request:
            description = self.APPROVAL_REQUIRED_OPERATIONS.get(
                operation_type, operation_type
            )
            request = self.request_approval(
                operation_type=operation_type,
                description=description,
                details=kwargs,
            )
            return {
                "allowed": False,
                "requires_approval": True,
                "request_id": request.request_id,
                "message": "Approval required. Request created.",
            }
        
        return {
            "allowed": False,
            "requires_approval": True,
            "request_id": None,
            "message": "Approval required. No pending request found.",
        }
    
    def get_approval_history(self) -> List[Dict[str, Any]]:
        """Get the full approval history."""
        return [r.to_dict() for r in self._approval_history]
    
    def enable_auto_approve(self, enable: bool = True) -> None:
        """Enable or disable auto-approve mode (testing only)."""
        self.auto_approve = enable
        self.logger.warning(f"Auto-approve mode {'enabled' if enable else 'disabled'}")