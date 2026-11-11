"""STRICT_PLUS effect taxonomy.

Every operation registered in UZEN declares the *kind of computational effect* it
has on the system. The policy engine reasons about effects, not operation names,
so a newly added operation cannot accidentally smuggle in gradient-based
training: an operation that needs autograd must declare `GRADIENT_UPDATE`, which
STRICT_PLUS denies unconditionally.

Declaring an effect is not sufficient by itself — the execution layer also
enforces a runtime guard (see `uzen.policies.strict_plus.engine` and
`uzen.runtimes.pytorch`), so a mislabelled operation still fails closed.
"""

from __future__ import annotations

from enum import Enum


class EngineEffect(str, Enum):
    """The computational effect of an operation."""

    READ = "read"                     # inspection / parsing only, no math mutation
    IO = "io"                         # artifact serialization / transport
    PARAM_TRANSFORM = "param_transform"           # deterministic closed-form math on tensors
    DERIVATIVE_FREE = "derivative_free_search"    # search without gradients (ES/CMA/SA/BO)
    FORWARD_ONLY = "forward_only"                 # forward execution, no parameter update
    ARCH_MUTATION = "arch_mutation"               # computational-graph change
    GRADIENT_UPDATE = "gradient_update"           # PROHIBITED
    OPTIMIZER_STATE = "optimizer_state"           # PROHIBITED
    DATA_TRAINING = "data_training"               # PROHIBITED
    AUDIT = "audit"                               # provenance / verification bookkeeping


#: Effects that are unconditionally prohibited by STRICT_PLUS.
PROHIBITED_EFFECTS: frozenset[EngineEffect] = frozenset(
    {EngineEffect.GRADIENT_UPDATE, EngineEffect.OPTIMIZER_STATE, EngineEffect.DATA_TRAINING}
)

#: Effects that are permitted, with the justification used in policy decisions.
PERMITTED_EFFECTS: dict[EngineEffect, str] = {
    EngineEffect.READ: "inspection and parsing are always permitted",
    EngineEffect.IO: "artifact import/export is always permitted",
    EngineEffect.PARAM_TRANSFORM: (
        "deterministic closed-form parameter transformations are permitted; "
        "no gradients are computed and no loss is minimized"
    ),
    EngineEffect.DERIVATIVE_FREE: (
        "derivative-free search over low-dimensional transformation parameters is permitted; "
        "tensors are never updated by a gradient signal"
    ),
    EngineEffect.FORWARD_ONLY: (
        "forward-pass experimentation is permitted; parameters are treated as constants"
    ),
    EngineEffect.ARCH_MUTATION: (
        "architecture mutation is permitted when the job scope is architecture engineering"
    ),
    EngineEffect.AUDIT: "provenance and verification bookkeeping is always permitted",
}


class TaskScope(str, Enum):
    """Authorized scope of a job. Scope is never silently expanded."""

    INSPECTION = "inspection"
    WEIGHT_GENERATION = "weight_generation"
    WEIGHT_MODIFICATION = "weight_modification"
    WEIGHT_RECONSTRUCTION = "weight_reconstruction"
    WEIGHT_OPTIMIZATION = "weight_optimization"
    WEIGHT_ARCHITECTURE_RESEARCH = "weight_architecture_research"
    MODEL_CREATION = "model_creation"
    MODEL_MODIFICATION = "model_modification"
    MODEL_RECONSTRUCTION = "model_reconstruction"
    MODEL_ARCHITECTURE_RESEARCH = "model_architecture_research"
    ARCHITECTURE_SYNTHESIS = "architecture_synthesis"

    @property
    def is_weight_scope(self) -> bool:
        return self in {
            TaskScope.WEIGHT_GENERATION,
            TaskScope.WEIGHT_MODIFICATION,
            TaskScope.WEIGHT_RECONSTRUCTION,
            TaskScope.WEIGHT_OPTIMIZATION,
            TaskScope.WEIGHT_ARCHITECTURE_RESEARCH,
        }

    @property
    def is_architecture_scope(self) -> bool:
        return self in {
            TaskScope.MODEL_CREATION,
            TaskScope.MODEL_MODIFICATION,
            TaskScope.MODEL_RECONSTRUCTION,
            TaskScope.MODEL_ARCHITECTURE_RESEARCH,
            TaskScope.ARCHITECTURE_SYNTHESIS,
        }


#: Which scopes authorise which effects.
SCOPE_EFFECTS: dict[TaskScope, frozenset[EngineEffect]] = {
    TaskScope.INSPECTION: frozenset(
        {EngineEffect.READ, EngineEffect.IO, EngineEffect.AUDIT, EngineEffect.FORWARD_ONLY}
    ),
    TaskScope.WEIGHT_GENERATION: frozenset(
        {EngineEffect.READ, EngineEffect.IO, EngineEffect.PARAM_TRANSFORM,
         EngineEffect.DERIVATIVE_FREE, EngineEffect.FORWARD_ONLY, EngineEffect.AUDIT}
    ),
    TaskScope.WEIGHT_MODIFICATION: frozenset(
        {EngineEffect.READ, EngineEffect.IO, EngineEffect.PARAM_TRANSFORM,
         EngineEffect.DERIVATIVE_FREE, EngineEffect.FORWARD_ONLY, EngineEffect.AUDIT}
    ),
    TaskScope.WEIGHT_RECONSTRUCTION: frozenset(
        {EngineEffect.READ, EngineEffect.IO, EngineEffect.PARAM_TRANSFORM,
         EngineEffect.DERIVATIVE_FREE, EngineEffect.FORWARD_ONLY, EngineEffect.AUDIT}
    ),
    TaskScope.WEIGHT_OPTIMIZATION: frozenset(
        {EngineEffect.READ, EngineEffect.IO, EngineEffect.PARAM_TRANSFORM,
         EngineEffect.DERIVATIVE_FREE, EngineEffect.FORWARD_ONLY, EngineEffect.AUDIT}
    ),
    TaskScope.WEIGHT_ARCHITECTURE_RESEARCH: frozenset(
        {EngineEffect.READ, EngineEffect.IO, EngineEffect.PARAM_TRANSFORM,
         EngineEffect.DERIVATIVE_FREE, EngineEffect.FORWARD_ONLY, EngineEffect.AUDIT}
    ),
    TaskScope.MODEL_CREATION: frozenset(
        {EngineEffect.READ, EngineEffect.IO, EngineEffect.PARAM_TRANSFORM,
         EngineEffect.ARCH_MUTATION, EngineEffect.FORWARD_ONLY, EngineEffect.AUDIT}
    ),
    TaskScope.MODEL_MODIFICATION: frozenset(
        {EngineEffect.READ, EngineEffect.IO, EngineEffect.PARAM_TRANSFORM,
         EngineEffect.ARCH_MUTATION, EngineEffect.FORWARD_ONLY, EngineEffect.AUDIT}
    ),
    TaskScope.MODEL_RECONSTRUCTION: frozenset(
        {EngineEffect.READ, EngineEffect.IO, EngineEffect.PARAM_TRANSFORM,
         EngineEffect.ARCH_MUTATION, EngineEffect.FORWARD_ONLY, EngineEffect.AUDIT}
    ),
    TaskScope.MODEL_ARCHITECTURE_RESEARCH: frozenset(
        {EngineEffect.READ, EngineEffect.IO, EngineEffect.PARAM_TRANSFORM,
         EngineEffect.ARCH_MUTATION, EngineEffect.FORWARD_ONLY, EngineEffect.AUDIT}
    ),
    TaskScope.ARCHITECTURE_SYNTHESIS: frozenset(
        {EngineEffect.READ, EngineEffect.IO, EngineEffect.PARAM_TRANSFORM,
         EngineEffect.ARCH_MUTATION, EngineEffect.FORWARD_ONLY, EngineEffect.AUDIT}
    ),
}


class Maturity(str, Enum):
    """Capability maturity. Nothing is reported as implemented unless tested."""

    IMPLEMENTED = "implemented"
    PROTOTYPE = "prototype"
    PLANNED = "planned"
    UNSUPPORTED = "unsupported"

    @property
    def executable(self) -> bool:
        return self in (Maturity.IMPLEMENTED, Maturity.PROTOTYPE)


class Origin(str, Enum):
    """Origin classification for every tensor in a produced artifact."""

    INHERITED = "inherited"
    EXTRACTED = "extracted"
    TRANSFORMED = "transformed"
    ANALYTICALLY_DERIVED = "analytically_derived"
    RANDOMLY_INITIALIZED = "randomly_initialized"
    SEARCH_SELECTED = "search_selected"
    RECONSTRUCTED = "reconstructed"
