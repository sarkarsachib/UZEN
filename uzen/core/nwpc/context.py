"""Execution context for weight operations.

The context carries the execution seed, the policy engine, job identity and a
scratch area, and provides the only approved way to write tensors: every write is
checked by the policy engine's mutation guard, which fails closed if a gradient
context is somehow active. It also records the operations performed so an
artifact's lineage is complete without extra bookkeeping in each operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from uzen.policies.strict_plus.effects import TaskScope
from uzen.policies.strict_plus.engine import StrictPlusEngine
from uzen.util.rng import generator


@dataclass
class OperationContext:
    scope: TaskScope
    seed: int
    engine: StrictPlusEngine
    operation: str
    job_id: str | None = None
    scratch: dict[str, Any] = field(default_factory=dict)
    writes: int = 0
    notes: list[str] = field(default_factory=list)

    def rng(self, *path: str | int) -> np.random.Generator:
        return generator(self.seed, self.operation, *path)

    def child_seed(self, *path: str | int) -> int:
        from uzen.util.rng import derive_seed

        return derive_seed(self.seed, self.operation, *path)

    def emit(self, array: np.ndarray, *, name: str) -> np.ndarray:
        """Register a produced tensor (policy-checked, must be finite)."""
        self.engine.guard_mutation(
            f"{self.operation}:{name}", gradient_active=_is_grad_tracking_tensor(array)
        )
        arr = np.ascontiguousarray(np.asarray(array))
        if arr.dtype.kind in "fc" and not np.isfinite(arr).all():
            raise ValueError(
                f"operation '{self.operation}' produced non-finite values for tensor '{name}'"
            )
        self.writes += 1
        return arr

    def note(self, message: str) -> None:
        self.notes.append(message)

    def audit_extras(self) -> dict[str, Any]:
        return {"writes": self.writes, "notes": list(self.notes)}


def _is_grad_tracking_tensor(array: Any) -> bool:
    """True only if this exact tensor is being autograd-tracked.

    STRICT_PLUS forbids parameter mutation inside a gradient tape. We therefore
    check the *tensor being written* rather than the process-wide autograd switch
    (which is enabled by default in PyTorch and would otherwise block legitimate
    NumPy-side engineering). The PyTorch runtime additionally runs every forward
    pass under ``inference_mode``, so no graph is ever recorded for UZEN's own
    computations.
    """
    requires_grad = getattr(array, "requires_grad", False)
    return bool(requires_grad) and type(array).__module__.startswith("torch")
