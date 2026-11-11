"""Weight Generation Engine (WGE).

Generates complete parameter packages from:

* an architecture family spec (structured random / analytic construction),
* a template package's shapes (structured random, constraint-matched),
* donor factors (exact algebraic reconstruction).

Every generated artifact is labelled with its true origin and seed. WGE makes no
claim that generated parameters are intelligent — only that they are valid,
reproducible parameter artifacts of the requested structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from uzen.core.nws.families import ArchitectureDefinition, build_architecture
from uzen.core.nws.tensors import WeightPackage
from uzen.core.nwpc.registry import OperationRecord, default_registry
from uzen.errors import InvalidOperationParameters
from uzen.policies.strict_plus.effects import TaskScope
from uzen.policies.strict_plus.engine import StrictPlusEngine, default_engine

METHODS = {
    "orthogonal": ("w.gen.structured", {"distribution": "orthogonal"}),
    "structured_random": ("w.gen.structured", {"distribution": "normal"}),
    "xavier": ("w.gen.structured", {"distribution": "xavier_uniform"}),
    "he": ("w.gen.structured", {"distribution": "he_normal"}),
    "constraint_matched": ("w.gen.constraint_matched", {}),
    "from_factors": ("w.gen.from_factors", {}),
    "low_rank": ("w.gen.low_rank_regeneration", {"rank": 8}),
}


@dataclass
class GenerationResult:
    package: WeightPackage
    record: OperationRecord | None
    definition: ArchitectureDefinition | None
    method: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "family": self.package.family,
            "num_tensors": len(self.package),
            "numel": self.package.numel(),
            "origins": {name: meta.origin.value for name, meta in self.package.metas.items()},
            "operation": self.record.to_dict() if self.record else None,
            "note": "generated parameters are not learned (see per-tensor origin labels)",
        }


class WeightGenerationEngine:
    def __init__(self, *, engine: StrictPlusEngine | None = None) -> None:
        self.engine = engine or default_engine()
        self.registry = default_registry()

    # ------------------------------------------------------------------ #

    def from_architecture(
        self,
        family: str,
        spec: Mapping[str, Any] | None = None,
        *,
        seed: int | str | None = 0,
        dtype: str = "float32",
        name: str | None = None,
    ) -> GenerationResult:
        definition = build_architecture(family, spec)
        package = definition.build_package(seed=_seed_int(seed), dtype=dtype, name=name)
        definition.graph.require_valid(package)
        self.engine.audit.append(
            "weights_generated",
            {
                "family": family,
                "spec": definition.spec,
                "seed": _seed_int(seed),
                "num_tensors": len(package),
                "numel": package.numel(),
                "origin": "structured/analytic initialization (not learned)",
            },
        )
        return GenerationResult(
            package=package,
            record=None,
            definition=definition,
            method="architecture_init",
            seed=_seed_int(seed),
        )

    def generate(
        self,
        template: WeightPackage,
        *,
        method: str = "orthogonal",
        params: Mapping[str, Any] | None = None,
        donors: Sequence[WeightPackage] = (),
        seed: int | str | None = 0,
        scope: TaskScope = TaskScope.WEIGHT_GENERATION,
    ) -> GenerationResult:
        if method not in METHODS:
            raise InvalidOperationParameters(
                f"unknown generation method '{method}'; available: {sorted(METHODS)}"
            )
        code, defaults = METHODS[method]
        merged = {**defaults, **dict(params or {})}
        inputs = [template, *donors]
        record = self.registry.invoke(
            code,
            inputs,
            merged,
            scope=scope,
            seed=seed,
            engine=self.engine,
        )
        return GenerationResult(
            package=record.output,
            record=record,
            definition=None,
            method=method,
            seed=record.seed,
        )

    def methods(self) -> dict[str, Any]:
        return {
            name: {
                "operation": code,
                "default_params": defaults,
                "requires_donor": name in ("constraint_matched", "from_factors", "low_rank"),
                "origin": "randomly_initialized" if name in ("orthogonal", "structured_random", "xavier", "he") else "analytically_derived",
            }
            for name, (code, defaults) in METHODS.items()
        }


def _seed_int(seed: int | str | None) -> int:
    from uzen.util.rng import parse_seed

    return parse_seed(seed)
