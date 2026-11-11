"""STRICT_PLUS policy engine and hash-chained audit log.

The policy engine is consulted at four independent layers:

1. **Operation registry** — an operation is only invocable if its declared
   effects are permitted for the session's scope.
2. **Planner** — a whole engineering plan is rejected if any stage carries a
   prohibited effect, even if the planner (including an LLM planner) proposed it.
3. **Tensor mutation interface** — the execution context refuses writes when a
   gradient tape / autograd mode is active.
4. **Artifact validation** — provenance records must carry the policy decision
   that authorised them; artifacts without a valid decision chain fail audit.

Every decision is appended to an append-only, hash-chained audit log so that
tampering with history is detectable.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from uzen.errors import PolicyViolation, ScopeViolation
from uzen.policies.strict_plus.effects import (
    PROHIBITED_EFFECTS,
    SCOPE_EFFECTS,
    EngineEffect,
    Maturity,
    TaskScope,
)
from uzen.util.canonical import canonical_bytes, sha256_bytes, utc_now_iso

STRICT_PLUS_VERSION = "1.0"


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    effect: EngineEffect
    scope: TaskScope
    reason: str
    subject: str
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "effect": self.effect.value,
            "scope": self.scope.value,
            "reason": self.reason,
            "subject": self.subject,
            "policy": f"STRICT_PLUS/{STRICT_PLUS_VERSION}",
            "timestamp": self.timestamp,
        }

    def require(self) -> None:
        if not self.allowed:
            raise PolicyViolation(self.reason, effect=self.effect.value, subject=self.subject)


class AuditLog:
    """Append-only hash-chained log of policy decisions and executions.

    Each entry embeds the SHA-256 of the previous entry's canonical bytes, so any
    modification of earlier history invalidates the chain from that point on.
    """

    GENESIS = "0" * 64

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self.path = Path(path) if path is not None else None
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._entries.append(json.loads(line))

    @property
    def head(self) -> str:
        return self._entries[-1]["entry_hash"] if self._entries else self.GENESIS

    def append(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            entry = {
                "seq": len(self._entries),
                "kind": kind,
                "payload": payload,
                "prev_hash": self.head,
                "timestamp": utc_now_iso(),
            }
            entry["entry_hash"] = sha256_bytes(canonical_bytes(entry))
            self._entries.append(entry)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            return entry

    def record_decision(self, decision: PolicyDecision) -> dict[str, Any]:
        return self.append("policy_decision", decision.to_dict())

    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def verify(self) -> tuple[bool, str | None]:
        prev = self.GENESIS
        for i, entry in enumerate(self._entries):
            body = {k: v for k, v in entry.items() if k != "entry_hash"}
            if entry.get("prev_hash") != prev:
                return False, f"entry {i} prev_hash mismatch"
            if sha256_bytes(canonical_bytes(body)) != entry.get("entry_hash"):
                return False, f"entry {i} hash mismatch"
            prev = entry["entry_hash"]
        return True, None


class StrictPlusEngine:
    """Central policy authority. Fail-closed by design."""

    def __init__(self, *, audit: AuditLog | None = None, strict: bool = True) -> None:
        self.audit = audit or AuditLog()
        self.strict = strict
        self._active_scopes: list[TaskScope] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # scope handling
    # ------------------------------------------------------------------ #

    @contextmanager
    def scope(self, scope: TaskScope) -> Iterator[TaskScope]:
        with self._lock:
            self._active_scopes.append(scope)
        try:
            yield scope
        finally:
            with self._lock:
                self._active_scopes.pop()

    @property
    def current_scope(self) -> TaskScope:
        with self._lock:
            return self._active_scopes[-1] if self._active_scopes else TaskScope.INSPECTION

    # ------------------------------------------------------------------ #
    # decisions
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        effect: EngineEffect,
        subject: str,
        *,
        scope: TaskScope | None = None,
        maturity: Maturity = Maturity.IMPLEMENTED,
        requirements: Sequence[str] = (),
    ) -> PolicyDecision:
        scope = scope or self.current_scope

        if effect in PROHIBITED_EFFECTS:
            decision = PolicyDecision(
                allowed=False,
                effect=effect,
                scope=scope,
                subject=subject,
                reason=(
                    f"STRICT_PLUS prohibits '{effect.value}': UZEN never performs conventional "
                    "training, fine-tuning, backpropagation, gradient descent or gradient-based "
                    "parameter updates. Use closed-form, task-arithmetic, alignment or "
                    "derivative-free alternatives."
                ),
            )
            self.audit.record_decision(decision)
            return decision

        allowed_effects = SCOPE_EFFECTS.get(scope, frozenset())
        if effect not in allowed_effects:
            decision = PolicyDecision(
                allowed=False,
                effect=effect,
                scope=scope,
                subject=subject,
                reason=(
                    f"effect '{effect.value}' is outside the authorized scope '{scope.value}'. "
                    "UZEN never silently expands the requested scope; submit a job whose "
                    "task_scope covers this operation."
                ),
            )
            self.audit.record_decision(decision)
            return decision

        if not maturity.executable:
            decision = PolicyDecision(
                allowed=False,
                effect=effect,
                scope=scope,
                subject=subject,
                reason=(
                    f"'{subject}' has maturity '{maturity.value}' and is not executable in this "
                    "build; UZEN will not simulate a result for an unimplemented capability."
                ),
            )
            self.audit.record_decision(decision)
            return decision

        decision = PolicyDecision(
            allowed=True,
            effect=effect,
            scope=scope,
            subject=subject,
            reason="permitted under STRICT_PLUS (non-training engineering operation)",
        )
        self.audit.record_decision(decision)
        self._check_requirements(requirements, subject, scope)
        return decision

    def _check_requirements(self, requirements: Sequence[str], subject: str, scope: TaskScope) -> None:
        banned = ("requires_grad", "backward", "autograd", "optimizer", "learning_rate", "epoch")
        for req in requirements:
            low = req.lower()
            for b in banned:
                if b in low:
                    raise PolicyViolation(
                        f"operation '{subject}' declares a requirement containing '{b}'; "
                        "STRICT_PLUS forbids gradient-based execution paths",
                        requirement=req,
                    )

    def authorize(
        self,
        effect: EngineEffect,
        subject: str,
        *,
        scope: TaskScope | None = None,
        maturity: Maturity = Maturity.IMPLEMENTED,
        requirements: Sequence[str] = (),
    ) -> PolicyDecision:
        decision = self.evaluate(
            effect, subject, scope=scope, maturity=maturity, requirements=requirements
        )
        decision.require()
        return decision

    def authorize_plan(
        self, stages: Iterable[tuple[str, EngineEffect, Maturity]], scope: TaskScope
    ) -> list[PolicyDecision]:
        """Authorize a complete engineering plan atomically (all-or-nothing)."""
        decisions: list[PolicyDecision] = []
        for subject, effect, maturity in stages:
            d = self.evaluate(effect, subject, scope=scope, maturity=maturity)
            decisions.append(d)
            if not d.allowed:
                self.audit.append(
                    "plan_rejected",
                    {"scope": scope.value, "rejected_stage": subject, "reason": d.reason},
                )
                raise PolicyViolation(
                    f"engineering plan rejected at stage '{subject}': {d.reason}",
                    stage=subject,
                    scope=scope.value,
                )
        self.audit.append(
            "plan_authorized",
            {"scope": scope.value, "stages": [s[0] for s in stages]},
        )
        return decisions

    # ------------------------------------------------------------------ #
    # runtime mutation guard
    # ------------------------------------------------------------------ #

    def guard_mutation(self, subject: str, *, gradient_active: bool, scope: TaskScope | None = None) -> None:
        """Called on every parameter write by the execution layer."""
        if gradient_active:
            d = self.evaluate(EngineEffect.GRADIENT_UPDATE, f"mutation:{subject}", scope=scope)
            raise PolicyViolation(
                "refusing parameter mutation while an autograd/gradient context is active: "
                f"{d.reason}",
                subject=subject,
            )

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #

    def describe(self) -> dict[str, Any]:
        return {
            "policy": "STRICT_PLUS",
            "version": STRICT_PLUS_VERSION,
            "strict": self.strict,
            "prohibited_effects": sorted(e.value for e in PROHIBITED_EFFECTS),
            "permitted_effects": sorted(e.value for e in EngineEffect if e not in PROHIBITED_EFFECTS),
            "scope_matrix": {
                scope.value: sorted(e.value for e in effects)
                for scope, effects in SCOPE_EFFECTS.items()
            },
            "statements": [
                "UZEN does not perform conventional model training.",
                "UZEN does not fine-tune models or run backpropagation.",
                "UZEN never applies gradient-based parameter updates.",
                "Random initialization is always labelled randomly_initialized.",
                "Original source weights are immutable; outputs are new content-addressed artifacts.",
                "An unimplemented capability is reported as planned/unsupported, never simulated.",
            ],
            "audit_head": self.audit.head,
            "audit_entries": len(self.audit.entries()),
        }


_default_engine: StrictPlusEngine | None = None
_default_lock = threading.Lock()


def default_engine() -> StrictPlusEngine:
    global _default_engine
    with _default_lock:
        if _default_engine is None:
            audit_path = os.environ.get("UZEN_AUDIT_LOG")
            _default_engine = StrictPlusEngine(
                audit=AuditLog(audit_path) if audit_path else AuditLog()
            )
        return _default_engine


def reset_default_engine(engine: StrictPlusEngine | None = None) -> None:
    """Test/embedding hook: replace the process-wide engine."""
    global _default_engine
    with _default_lock:
        _default_engine = engine
