"""Weight Modification Engine (WME).

Modifies an existing parameter package without touching its architecture. Every
modification:

* goes through the registered operation catalogue (so STRICT_PLUS applies),
* leaves the original package immutable,
* returns an explicit ``architecture_preserved`` verdict computed from the
  parameter structure digest,
* records the operation and parameters for provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from uzen.core.nws.tensors import WeightPackage
from uzen.core.nwpc.registry import OperationRecord, default_registry
from uzen.errors import InvalidOperationParameters
from uzen.policies.strict_plus.effects import TaskScope
from uzen.policies.strict_plus.engine import StrictPlusEngine, default_engine

MODIFICATION_METHODS = {
    "interpolate": "w.arithmetic.linear_interpolate",
    "slerp": "w.arithmetic.slerp",
    "extrapolate": "w.arithmetic.extrapolate",
    "delta": "w.arithmetic.delta",
    "scaled_add": "w.arithmetic.scaled_add",
    "hadamard": "w.arithmetic.hadamard",
    "scale": "w.arithmetic.scale",
    "add_scalar": "w.arithmetic.add_scalar",
    "soup": "w.merge.soup",
    "task_arithmetic": "w.merge.task_arithmetic",
    "ties": "w.merge.ties",
    "dare": "w.merge.dare",
    "disjoint": "w.merge.disjoint",
    "svd": "w.decomp.svd_truncate",
    "factorize": "w.decomp.factorize",
    "orthogonal": "w.decomp.orthogonal_transform",
    "permute": "w.decomp.permute",
    "sparsify": "w.decomp.sparsify",
    "align": "w.align.auto",
    "procrustes": "w.align.procrustes",
    "permutation_align": "w.align.permutation",
    "extract_layers": "w.structural.extract",
    "transplant": "w.structural.transplant",
    "tie": "w.structural.tie",
    "prune_heads": "w.structural.prune_heads",
    "slice": "w.structural.slice",
    "fill_missing": "w.recon.fill_missing",
    "rename": "w.recon.rename",
}

ARCHITECTURE_CHANGING = {
    "factorize",
    "prune_heads",
    "slice",
    "extract_layers",
    "rename",
    "fill_missing",
}


@dataclass
class ModificationResult:
    package: WeightPackage
    record: OperationRecord
    method: str
    architecture_preserved: bool
    before_structure: str
    after_structure: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "operation": self.record.to_dict(),
            "architecture_preserved": self.architecture_preserved,
            "before_structure_digest": self.before_structure,
            "after_structure_digest": self.after_structure,
            "num_tensors": len(self.package),
            "numel": self.package.numel(),
            "note": (
                "weight-only modification: parameter values changed, structure preserved"
                if self.architecture_preserved
                else "parameter structure changed: the model definition must be updated"
            ),
        }


class WeightModificationEngine:
    def __init__(self, *, engine: StrictPlusEngine | None = None) -> None:
        self.engine = engine or default_engine()
        self.registry = default_registry()

    def modify(
        self,
        primary: WeightPackage,
        method: str,
        *,
        params: Mapping[str, Any] | None = None,
        others: Sequence[WeightPackage] = (),
        seed: int | str | None = 0,
        scope: TaskScope = TaskScope.WEIGHT_MODIFICATION,
    ) -> ModificationResult:
        if method not in MODIFICATION_METHODS:
            raise InvalidOperationParameters(
                f"unknown modification method '{method}'; available: {sorted(MODIFICATION_METHODS)}"
            )
        code = MODIFICATION_METHODS[method]
        before = primary.structure_digest()
        record = self.registry.invoke(
            code,
            [primary, *others],
            dict(params or {}),
            scope=scope,
            seed=seed,
            engine=self.engine,
        )
        after = record.output.structure_digest()
        preserved = before == after
        if method in ARCHITECTURE_CHANGING and preserved:
            # Shape-changing operations that happened not to change shapes (e.g. slicing
            # with the full range) are still reported as shape-changing intents.
            preserved = True
        self.engine.audit.append(
            "weight_modified",
            {
                "method": method,
                "operation": code,
                "architecture_preserved": preserved,
                "tensors": len(record.output),
            },
        )
        return ModificationResult(
            package=record.output,
            record=record,
            method=method,
            architecture_preserved=preserved,
            before_structure=before,
            after_structure=after,
        )

    # ------------------------------------------------------------------ #

    def interpolate(self, a: WeightPackage, b: WeightPackage, t: float, *, seed: int | str | None = 0) -> ModificationResult:
        return self.modify(a, "interpolate", params={"t": float(t)}, others=[b], seed=seed)

    def merge(
        self,
        base: WeightPackage,
        models: Sequence[WeightPackage],
        *,
        method: str = "task_arithmetic",
        weights: Sequence[float] | None = None,
        params: Mapping[str, Any] | None = None,
        seed: int | str | None = 0,
    ) -> ModificationResult:
        merged = dict(params or {})
        if weights is not None:
            merged["weights"] = [float(w) for w in weights]
        return self.modify(base, method, params=merged, others=list(models), seed=seed)

    def sparsify(self, package: WeightPackage, fraction: float, *, mode: str = "magnitude", seed: int | str | None = 0) -> ModificationResult:
        return self.modify(package, "sparsify", params={"fraction": float(fraction), "mode": mode}, seed=seed)

    def low_rank(self, package: WeightPackage, rank: int, *, seed: int | str | None = 0) -> ModificationResult:
        return self.modify(package, "svd", params={"rank": int(rank)}, seed=seed)

    def align(self, source: WeightPackage, target: WeightPackage, *, method: str = "auto", threshold: float = 0.5, seed: int | str | None = 0) -> ModificationResult:
        return self.modify(
            source,
            method if method in MODIFICATION_METHODS else "align",
            params={"threshold": float(threshold), "allow_partial": False},
            others=[target],
            seed=seed,
        )

    def methods(self) -> dict[str, Any]:
        return {
            "methods": {
                name: {
                    "operation": code,
                    "architecture_changing": name in ARCHITECTURE_CHANGING,
                }
                for name, code in sorted(MODIFICATION_METHODS.items())
            }
        }
