"""Weight Architecture Research (WAR).

Investigates alternative *parameter organizations* — low-rank factorizations,
sparsification schemes, orthogonal basis changes, parameter tying — and reports
measured storage/size/error trade-offs. This is research output: a table of real
measurements plus artifacts, not a promise that a scheme is better for a task.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nws.tensors import WeightPackage
from uzen.core.nwpc.registry import default_registry
from uzen.errors import InvalidOperationParameters
from uzen.policies.strict_plus.effects import TaskScope
from uzen.policies.strict_plus.engine import StrictPlusEngine, default_engine


@dataclass
class ResearchExperiment:
    name: str
    operation: str
    params: dict[str, Any]
    ok: bool
    num_tensors: int = 0
    numel: int = 0
    stored_numel: int = 0
    relative_error: float | None = None
    spectral_energy: float | None = None
    latency_ms: float | None = None
    problems: list[str] = field(default_factory=list)
    output_package: WeightPackage | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.name,
            "operation": self.operation,
            "params": self.params,
            "ok": self.ok,
            "num_tensors": self.num_tensors,
            "numel": self.numel,
            "stored_numel": self.stored_numel,
            "compression_ratio": round(self.numel / self.stored_numel, 6) if self.stored_numel else None,
            "relative_error": None if self.relative_error is None else round(self.relative_error, 8),
            "spectral_energy": None if self.spectral_energy is None else round(self.spectral_energy, 8),
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 3),
            "problems": self.problems,
            "note": "shape-preserving schemes keep the architecture; factorized schemes require graph rebuild",
        }


class WeightArchitectureResearch:
    def __init__(self, *, engine: StrictPlusEngine | None = None) -> None:
        self.engine = engine or default_engine()
        self.registry = default_registry()

    def default_suite(self, package: WeightPackage) -> list[dict[str, Any]]:
        ranks = sorted({1, 2, 4, 8, 16, max(1, min(package.numel() // 1024, 64))})
        suite: list[dict[str, Any]] = [
            {"name": "baseline", "operation": "identity", "params": {}},
            {"name": "orthogonal_qr", "operation": "w.decomp.orthogonal_transform", "params": {"method": "qr", "side": "right"}},
            {"name": "sparsify_50", "operation": "w.decomp.sparsify", "params": {"fraction": 0.5, "mode": "magnitude"}},
            {"name": "sparsify_90", "operation": "w.decomp.sparsify", "params": {"fraction": 0.1, "mode": "magnitude"}},
        ]
        for rank in ranks:
            suite.append({"name": f"svd_r{rank}", "operation": "w.decomp.svd_truncate", "params": {"rank": rank}})
        for rank in ranks[:3]:
            suite.append({"name": f"factorize_r{rank}", "operation": "w.decomp.factorize", "params": {"rank": rank}})
        return suite

    def run(
        self,
        package: WeightPackage,
        *,
        experiments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | str | None = 0,
        scope: TaskScope = TaskScope.WEIGHT_ARCHITECTURE_RESEARCH,
    ) -> dict[str, Any]:
        suite = list(experiments or self.default_suite(package))
        if not suite:
            raise InvalidOperationParameters("no experiments supplied")
        results: list[ResearchExperiment] = []
        for entry in suite:
            name = str(entry.get("name", entry.get("operation", "experiment")))
            operation = str(entry.get("operation", ""))
            params = dict(entry.get("params") or {})
            if operation == "identity":
                results.append(
                    ResearchExperiment(
                        name=name,
                        operation="identity",
                        params={},
                        ok=True,
                        num_tensors=len(package),
                        numel=package.numel(),
                        stored_numel=package.numel(),
                        relative_error=0.0,
                        spectral_energy=1.0,
                        latency_ms=None,
                        output_package=package,
                    )
                )
                continue
            try:
                start = time.perf_counter()
                record = self.registry.invoke(
                    operation, [package], params, scope=scope, seed=seed, engine=self.engine
                )
                elapsed = (time.perf_counter() - start) * 1000.0
                out = record.output
                experiment = ResearchExperiment(
                    name=name,
                    operation=operation,
                    params=params,
                    ok=True,
                    num_tensors=len(out),
                    numel=package.numel(),
                    stored_numel=out.numel(),
                    latency_ms=elapsed,
                    output_package=out,
                )
                error, energy = self._compare(package, out)
                experiment.relative_error = error
                experiment.spectral_energy = energy
                results.append(experiment)
            except Exception as exc:
                results.append(
                    ResearchExperiment(
                        name=name,
                        operation=operation,
                        params=params,
                        ok=False,
                        problems=[f"{type(exc).__name__}: {exc}"],
                    )
                )
        ok_results = [r for r in results if r.ok]
        best_by_error = min(
            (r for r in ok_results if r.relative_error is not None),
            key=lambda r: (r.relative_error or 0.0),
            default=None,
        )
        best_by_size = min(
            (r for r in ok_results if r.stored_numel),
            key=lambda r: r.stored_numel,
            default=None,
        )
        return {
            "family": package.family,
            "num_tensors": len(package),
            "numel": package.numel(),
            "experiments": [r.to_dict() for r in results],
            "best_fidelity": best_by_error.name if best_by_error else None,
            "smallest": best_by_size.name if best_by_size else None,
            "measurement_notes": [
                "stored_numel counts the tensor elements an artifact would hold",
                "relative_error is ||candidate - baseline||_F / ||baseline||_F over shared tensors",
                "latency_ms measures the transformation, not inference",
                "factorized schemes change shapes and require a rebuilt graph to execute",
            ],
            "conclusion_policy": (
                "these are measurements of representation trade-offs; no claim is made "
                "that any scheme improves downstream task quality"
            ),
        }

    def _compare(self, baseline: WeightPackage, candidate: WeightPackage) -> tuple[float | None, float | None]:
        shared = [n for n in candidate.names() if n in baseline and candidate.tensors[n].shape == baseline.tensors[n].shape]
        if not shared:
            return None, None
        num = 0.0
        den = 0.0
        for name in shared:
            b = baseline.tensors[name].astype(np.float64)
            c = candidate.tensors[name].astype(np.float64)
            num += float(np.sum((b - c) ** 2))
            den += float(np.sum(b**2))
        error = float(np.sqrt(num / den)) if den > 0 else None
        energy = float(1.0 - (num / den)) if den > 0 else None
        return error, energy
