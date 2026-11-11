"""Typed error taxonomy for UZEN.

Every failure mode in the system has an explicit, machine-readable code. The
system never manufactures a successful result: when a requested operation
cannot be carried out defensibly the caller receives one of these errors with a
stable ``code`` so that agents, CLI scripts and the REST/MCP layers can react
programmatically.
"""

from __future__ import annotations

from typing import Any


class UzenError(Exception):
    """Base class for all UZEN errors."""

    code: str = "uzen.error"
    http_status: int = 500

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:  # pragma: no cover - formatting only
        if not self.details:
            return f"[{self.code}] {self.message}"
        return f"[{self.code}] {self.message} ({self.details})"


class PolicyViolation(UzenError):
    """STRICT_PLUS rejected the operation or plan."""

    code = "policy.strict_plus.violation"
    http_status = 403


class ScopeViolation(UzenError):
    """The requested action exceeds (or contradicts) the authorized task scope."""

    code = "policy.scope.violation"
    http_status = 403


class OperationNotRegistered(UzenError):
    code = "operation.not_registered"
    http_status = 404


class OperationUnavailable(UzenError):
    """Operation exists in the catalogue but is not executable in this build."""

    code = "operation.unavailable"
    http_status = 501


class InvalidOperationParameters(UzenError):
    code = "operation.invalid_parameters"
    http_status = 422


class ShapeMismatch(UzenError):
    code = "tensor.shape_mismatch"
    http_status = 422


class DTypeMismatch(UzenError):
    code = "tensor.dtype_mismatch"
    http_status = 422


class TensorInvalid(UzenError):
    """Non-finite values, impossible dtype, or corrupted buffer."""

    code = "tensor.invalid"
    http_status = 422


class NoValidAlignment(UzenError):
    """No defensible mathematical correspondence exists between parameters."""

    code = "npa.no_valid_alignment"
    http_status = 409


class NoValidStrategy(UzenError):
    code = "ane.no_valid_strategy"
    http_status = 409


class ArtifactNotFound(UzenError):
    code = "artifact.not_found"
    http_status = 404


class ArtifactIntegrityError(UzenError):
    code = "artifact.integrity"
    http_status = 500


class ArtifactImmutable(UzenError):
    """Attempted to overwrite an existing content-addressed artifact."""

    code = "artifact.immutable"
    http_status = 409


class UnsupportedFormat(UzenError):
    code = "format.unsupported"
    http_status = 415


class UnsupportedOperator(UzenError):
    code = "nir.unsupported_operator"
    http_status = 422


class GraphValidationError(UzenError):
    code = "nir.invalid_graph"
    http_status = 422


class ProgramInvalid(UzenError):
    code = "wpl.invalid_program"
    http_status = 422


class ValidationFailed(UzenError):
    code = "vel.validation_failed"
    http_status = 422


class BudgetExhausted(UzenError):
    code = "ane.budget_exhausted"
    http_status = 429


class WorkerAuthError(UzenError):
    code = "worker.auth_failed"
    http_status = 401


class WorkerCapabilityError(UzenError):
    code = "worker.capability"
    http_status = 409


class JobNotFound(UzenError):
    code = "job.not_found"
    http_status = 404


class JobStateError(UzenError):
    code = "job.invalid_state"
    http_status = 409


class IdempotencyConflict(UzenError):
    code = "job.idempotency_conflict"
    http_status = 409


class RuntimeUnavailable(UzenError):
    """An optional execution backend (e.g. PyTorch) is not installed."""

    code = "runtime.unavailable"
    http_status = 501


class NotImplementedYet(UzenError):
    """Capability is designed but explicitly not implemented in this build."""

    code = "capability.planned"
    http_status = 501
