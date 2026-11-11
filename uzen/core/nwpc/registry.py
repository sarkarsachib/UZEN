"""The operation registry: the only path through which parameters change.

Design rules
------------
* Every parameter-changing operation is **registered**, with a code, an effect,
  a maturity level, a scope set, typed parameters and explicit mathematical
  assumptions.
* The registry is the single choke point that consults STRICT_PLUS. An AI planner
  may propose anything; the registry refuses anything unregistered or prohibited.
* Operations are deterministic given inputs, parameters and seed. Randomness is
  always seeded and the seed is returned for provenance.
* Operations never mutate their inputs: they return a new package. Original
  source weights are therefore immutable by construction.

Two invocation shapes exist:

* :meth:`OperationRegistry.invoke` — data transformation over weight packages.
* :meth:`OperationRegistry.invoke_search` — derivative-free search over a
  caller-supplied objective (low-dimensional transformation parameters only).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from uzen.core.nws.tensors import WeightPackage
from uzen.core.nwpc.context import OperationContext
from uzen.errors import InvalidOperationParameters, OperationNotRegistered, OperationUnavailable
from uzen.policies.strict_plus.effects import EngineEffect, Maturity, Origin, TaskScope
from uzen.policies.strict_plus.engine import StrictPlusEngine, default_engine
from uzen.util.rng import parse_seed


@dataclass(frozen=True)
class ParamSpec:
    type: str  # float | int | bool | str | list[str] | list[float] | list[int] | dict
    required: bool = False
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None
    description: str = ""

    def coerce(self, name: str, value: Any) -> Any:
        t = self.type
        if value is None and not self.required:
            # Optional parameters default to None and mean "unset" (e.g. select).
            return None
        if t == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise InvalidOperationParameters(f"parameter '{name}' must be a number, got {value!r}")
            out: Any = float(value)
        elif t == "int":
            if isinstance(value, bool):
                raise InvalidOperationParameters(f"parameter '{name}' must be an integer")
            if isinstance(value, (np.integer,)):
                out = int(value)
            elif isinstance(value, float) and value.is_integer():
                out = int(value)
            elif isinstance(value, (int, np.integer)):
                out = int(value)
            else:
                raise InvalidOperationParameters(f"parameter '{name}' must be an integer, got {value!r}")
        elif t == "bool":
            if not isinstance(value, bool):
                raise InvalidOperationParameters(f"parameter '{name}' must be a bool, got {value!r}")
            out = value
        elif t == "str":
            if not isinstance(value, str):
                raise InvalidOperationParameters(f"parameter '{name}' must be a string, got {value!r}")
            out = value
        elif t in ("list[str]", "list[float]", "list[int]"):
            if not isinstance(value, (list, tuple)):
                raise InvalidOperationParameters(f"parameter '{name}' must be a list")
            inner = t[5:-1]
            if inner == "str":
                out = [str(v) for v in value]
            elif inner == "float":
                out = [float(v) for v in value]
            else:
                out = [int(v) for v in value]
        elif t == "dict":
            if not isinstance(value, Mapping):
                raise InvalidOperationParameters(f"parameter '{name}' must be an object")
            out = dict(value)
        else:  # pragma: no cover - developer error
            raise InvalidOperationParameters(f"unknown parameter type '{t}'")
        if self.minimum is not None and isinstance(out, (int, float)) and not isinstance(out, bool):
            if out < self.minimum:
                raise InvalidOperationParameters(
                    f"parameter '{name}'={out} is below minimum {self.minimum}"
                )
        if self.maximum is not None and isinstance(out, (int, float)) and not isinstance(out, bool):
            if out > self.maximum:
                raise InvalidOperationParameters(
                    f"parameter '{name}'={out} is above maximum {self.maximum}"
                )
        if self.choices is not None and isinstance(out, str) and out not in self.choices:
            raise InvalidOperationParameters(
                f"parameter '{name}' must be one of {list(self.choices)}, got '{out}'"
            )
        return out


@dataclass
class OperationOutcome:
    """What an operation produced."""

    tensors: dict[str, np.ndarray]
    replaced: set[str] = field(default_factory=set)
    dropped: set[str] = field(default_factory=set)
    added: set[str] = field(default_factory=set)
    details: dict[str, Any] = field(default_factory=dict)
    origin: Origin = Origin.TRANSFORMED
    alignment: dict[str, Any] | None = None
    family_spec: dict[str, Any] | None = None


@dataclass
class SearchRequest:
    objective: Callable[[np.ndarray], float]
    bounds: list[tuple[float, float]]
    budget: int
    seed: int
    maximize: bool = True
    options: dict[str, Any] = field(default_factory=dict)
    # Additional objectives for Pareto search (all maximised). Scalar algorithms ignore
    # them; NSGA-II requires at least one.
    extra_objectives: tuple[Callable[[np.ndarray], float], ...] = ()


@dataclass
class SearchResult:
    best_params: np.ndarray
    best_value: float
    history: list[dict[str, Any]]
    evaluations: int
    converged: bool
    algorithm: str
    details: dict[str, Any] = field(default_factory=dict)
    # Non-dominated candidate set for multi-objective search (empty for scalar algorithms).
    pareto_front: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "best_params": [float(v) for v in self.best_params],
            "best_value": float(self.best_value),
            "evaluations": self.evaluations,
            "converged": self.converged,
            "details": self.details,
            "history_len": len(self.history),
            "pareto_front": self.pareto_front,
        }


@dataclass
class OperationRecord:
    code: str
    title: str
    output: WeightPackage
    params_used: dict[str, Any]
    seed: int
    effect: EngineEffect
    origin: Origin
    details: dict[str, Any]
    alignment: dict[str, Any] | None
    policy: list[dict[str, Any]]
    parents: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "params": self.params_used,
            "seed": self.seed,
            "effect": self.effect.value,
            "origin": self.origin.value,
            "details": self.details,
            "alignment": self.alignment,
            "parents": self.parents,
        }


@dataclass
class OperationSpec:
    code: str
    title: str
    family: str
    effect: EngineEffect
    # ``fn`` is None for derivative-free search algorithms: they have no
    # data-transformation semantics and are invoked through ``invoke_search``.
    fn: Callable[[OperationContext, list[WeightPackage], dict[str, Any]], OperationOutcome] | None
    description: str = ""
    maturity: Maturity = Maturity.IMPLEMENTED
    scopes: frozenset[TaskScope] = frozenset(
        {
            TaskScope.WEIGHT_GENERATION,
            TaskScope.WEIGHT_MODIFICATION,
            TaskScope.WEIGHT_RECONSTRUCTION,
            TaskScope.WEIGHT_OPTIMIZATION,
            TaskScope.MODEL_CREATION,
            TaskScope.MODEL_RECONSTRUCTION,
        }
    )
    params: dict[str, ParamSpec] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    requires_alignment: bool = False
    mutates_architecture: bool = False
    origin: Origin = Origin.TRANSFORMED
    min_inputs: int = 1
    max_inputs: int = 16
    search: Callable[[SearchRequest], SearchResult] | None = None
    observable: bool = True

    def validate_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        unknown = set(params) - set(self.params)
        if unknown:
            raise InvalidOperationParameters(
                f"operation '{self.code}' does not accept parameters {sorted(unknown)}; "
                f"accepted: {sorted(self.params)}"
            )
        out: dict[str, Any] = {}
        missing: list[str] = []
        for name, spec in self.params.items():
            if name in params:
                out[name] = spec.coerce(name, params[name])
            elif spec.required:
                missing.append(name)
            else:
                out[name] = spec.default
        if missing:
            raise InvalidOperationParameters(
                f"operation '{self.code}' is missing required parameters {missing}"
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "family": self.family,
            "description": self.description,
            "effect": self.effect.value,
            "maturity": self.maturity.value,
            "scopes": sorted(s.value for s in self.scopes),
            "assumptions": list(self.assumptions),
            "requires_alignment": self.requires_alignment,
            "mutates_architecture": self.mutates_architecture,
            "origin": self.origin.value,
            "params": {
                k: {
                    "type": v.type,
                    "required": v.required,
                    "default": v.default,
                    "minimum": v.minimum,
                    "maximum": v.maximum,
                    "choices": list(v.choices) if v.choices else None,
                    "description": v.description,
                }
                for k, v in self.params.items()
            },
        }


class OperationRegistry:
    def __init__(self, engine: StrictPlusEngine | None = None) -> None:
        self._ops: dict[str, OperationSpec] = {}
        self._lock = threading.RLock()
        self.engine = engine or default_engine()

    # ------------------------------------------------------------------ #
    # registration
    # ------------------------------------------------------------------ #

    def register(self, spec: OperationSpec) -> OperationSpec:
        with self._lock:
            if spec.code in self._ops:
                raise ValueError(f"operation '{spec.code}' is already registered")
            if not spec.code.startswith("w."):
                raise ValueError(f"operation code '{spec.code}' must start with 'w.'")
            self._ops[spec.code] = spec
        return spec

    def get(self, code: str) -> OperationSpec:
        with self._lock:
            try:
                return self._ops[code]
            except KeyError as exc:
                close = [c for c in self._ops if code.split(".")[-1] in c]
                raise OperationNotRegistered(
                    f"operation '{code}' is not registered; "
                    f"did you mean {close[:5]}?" if close else f"operation '{code}' is not registered"
                ) from exc

    def all(self) -> list[OperationSpec]:
        with self._lock:
            return sorted(self._ops.values(), key=lambda s: s.code)

    def catalogue(self, *, family: str | None = None, executable_only: bool = False) -> list[dict[str, Any]]:
        items = self.all()
        if family:
            items = [i for i in items if i.family == family]
        if executable_only:
            items = [i for i in items if i.maturity.executable]
        return [i.to_dict() for i in items]

    def families(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for spec in self.all():
            out[spec.family] = out.get(spec.family, 0) + 1
        return out

    # ------------------------------------------------------------------ #
    # invocation
    # ------------------------------------------------------------------ #

    def invoke(
        self,
        code: str,
        inputs: Sequence[WeightPackage],
        params: Mapping[str, Any] | None = None,
        *,
        scope: TaskScope,
        seed: int | str | None = None,
        engine: StrictPlusEngine | None = None,
        job_id: str | None = None,
    ) -> OperationRecord:
        spec = self.get(code)
        engine = engine or self.engine

        if not spec.maturity.executable:
            raise OperationUnavailable(
                f"operation '{code}' has maturity '{spec.maturity.value}' and is not executable "
                "in this build; UZEN reports planned capabilities rather than simulating them",
                operation=code,
            )
        if spec.fn is None:
            raise OperationUnavailable(
                f"operation '{code}' is a derivative-free search algorithm; invoke it through "
                "invoke_search with a SearchRequest rather than as a data transformation",
                operation=code,
            )
        if scope not in spec.scopes:
            raise OperationUnavailable(
                f"operation '{code}' is only available in scopes "
                f"{sorted(s.value for s in spec.scopes)}; requested '{scope.value}'",
                operation=code,
            )
        if len(inputs) < spec.min_inputs or len(inputs) > spec.max_inputs:
            raise InvalidOperationParameters(
                f"operation '{code}' accepts {spec.min_inputs}..{spec.max_inputs} input packages, "
                f"got {len(inputs)}"
            )
        validated = spec.validate_params(params or {})

        decision = engine.authorize(spec.effect, spec.code, scope=scope, maturity=spec.maturity)
        seed_value = parse_seed(seed)

        ctx = OperationContext(
            scope=scope,
            seed=seed_value,
            engine=engine,
            job_id=job_id,
            operation=spec.code,
        )
        outcome = spec.fn(ctx, list(inputs), validated)

        if not isinstance(outcome, OperationOutcome):
            raise InvalidOperationParameters(
                f"operation '{code}' returned {type(outcome).__name__}, expected OperationOutcome"
            )
        output = _compose_output(inputs[0], outcome, origin=spec.origin)

        record = OperationRecord(
            code=spec.code,
            title=spec.title,
            output=output,
            params_used=validated,
            seed=seed_value,
            effect=spec.effect,
            origin=spec.origin,
            details=outcome.details,
            alignment=outcome.alignment,
            policy=[decision.to_dict()],
            parents=[p.source_artifact_id for p in inputs if p.source_artifact_id],
        )
        engine.audit.append(
            "operation_executed",
            {
                "code": spec.code,
                "scope": scope.value,
                "seed": seed_value,
                "tensors_in": len(inputs[0]),
                "tensors_out": len(output),
                "params": _jsonable(validated),
            },
        )
        return record

    def invoke_search(
        self,
        code: str,
        request: SearchRequest,
        *,
        scope: TaskScope,
        engine: StrictPlusEngine | None = None,
    ) -> tuple[SearchResult, dict[str, Any]]:
        spec = self.get(code)
        if spec.search is None:
            raise OperationUnavailable(
                f"operation '{code}' is not a search algorithm", operation=code
            )
        if not spec.maturity.executable:
            raise OperationUnavailable(
                f"search algorithm '{code}' has maturity '{spec.maturity.value}'", operation=code
            )
        if scope not in spec.scopes:
            raise OperationUnavailable(
                f"search '{code}' is not available in scope '{scope.value}'", operation=code
            )
        if request.budget <= 0:
            raise InvalidOperationParameters("search budget must be positive")
        for i, (lo, hi) in enumerate(request.bounds):
            if lo >= hi:
                raise InvalidOperationParameters(f"bound {i} is empty: [{lo}, {hi}]")
        engine = engine or self.engine
        decision = engine.authorize(spec.effect, spec.code, scope=scope, maturity=spec.maturity)
        result = spec.search(request)
        engine.audit.append(
            "search_executed",
            {
                "code": spec.code,
                "scope": scope.value,
                "budget": request.budget,
                "evaluations": result.evaluations,
                "best_value": result.best_value,
                "dims": len(request.bounds),
            },
        )
        return result, decision.to_dict()


def _compose_output(primary: WeightPackage, outcome: OperationOutcome, *, origin: Origin) -> WeightPackage:
    """Merge an operation outcome back into a package with honest metadata."""
    tensors = dict(primary.tensors)
    metas = dict(primary.metas)
    for name in outcome.dropped:
        tensors.pop(name, None)
        metas.pop(name, None)
    for name, arr in outcome.tensors.items():
        arr = np.ascontiguousarray(arr)
        tensors[name] = arr
        meta = metas.get(name)
        origin_use = origin if name in outcome.replaced or name in outcome.added else (
            meta.origin if meta else Origin.INHERITED
        )
        from uzen.core.nws.dtypes import dtype_name
        from uzen.core.nws.roles import infer_role

        from uzen.core.nws.tensors import TensorMeta

        meta_cls = type(meta) if meta is not None else TensorMeta
        metas[name] = meta_cls(
            name=name,
            shape=tuple(int(d) for d in arr.shape),
            dtype=dtype_name(arr.dtype),
            role=meta.role if meta else infer_role(name, arr.shape),
            origin=origin_use,
            source_artifact_id=primary.source_artifact_id,
            ops=(meta.ops if meta else ()) + (("op:" + str(outcome.details.get("op", "transform")),)),
            note=str(outcome.details.get("note", "")),
        )
    pkg = WeightPackage(
        tensors=tensors,
        metas=metas,
        name=primary.name,
        family=primary.family,
        family_spec=dict(outcome.family_spec if outcome.family_spec is not None else primary.family_spec),
        metadata={**primary.metadata, "last_operation": outcome.details.get("op", "")},
        source_artifact_id=primary.source_artifact_id,
    )
    return pkg


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


_registry: OperationRegistry | None = None
_registry_lock = threading.Lock()


def default_registry() -> OperationRegistry:
    """Return the process-wide registry, importing all operation modules once."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = OperationRegistry()
        reg = _registry
    if not reg.all():
        import uzen.operations  # noqa: F401  (registers operations)

    return reg


def reset_registry(registry: OperationRegistry | None = None) -> None:
    global _registry
    with _registry_lock:
        _registry = registry


def register_operation(**kwargs: Any) -> Callable[[Callable[..., OperationOutcome]], Callable[..., OperationOutcome]]:
    """Decorator registering an operation implementation on the default registry."""

    def decorator(fn: Callable[..., OperationOutcome]) -> Callable[..., OperationOutcome]:
        default_registry().register(OperationSpec(fn=fn, **kwargs))  # type: ignore[arg-type]
        return fn

    return decorator
