"""Model Architecture Research (MAR).

Explores alternative computational graphs and reports measured properties: node and
parameter counts, forward-pass latency, activation footprint, and structural
validity. Candidate weights are *structured random* initializations, so reported
metrics are architecture-level (cost/shape) — never task quality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nws.families import ArchitectureDefinition
from uzen.core.nase.synth import ArchitectureSynthesizer
from uzen.errors import InvalidOperationParameters
from uzen.policies.strict_plus.effects import TaskScope
from uzen.policies.strict_plus.engine import StrictPlusEngine, default_engine


@dataclass
class ArchitectureCandidate:
    name: str
    spec: dict[str, Any]
    ok: bool
    num_nodes: int = 0
    num_parameters: int = 0
    parameter_numel: int = 0
    latency_ms: float | None = None
    output_shapes: dict[str, list[int]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    definition: ArchitectureDefinition | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.name,
            "spec": self.spec,
            "ok": self.ok,
            "num_nodes": self.num_nodes,
            "num_parameters": self.num_parameters,
            "parameter_numel": self.parameter_numel,
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 3),
            "output_shapes": self.output_shapes,
            "problems": self.problems,
        }


class ModelArchitectureResearch:
    def __init__(self, *, engine: StrictPlusEngine | None = None) -> None:
        self.engine = engine or default_engine()
        self.synth = ArchitectureSynthesizer()

    def default_variants(self, family: str, base_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
        variants: list[dict[str, Any]] = [{"name": "baseline", **dict(base_spec), "family": family}]
        if family == "transformer_decoder":
            d_model = int(base_spec.get("d_model", 32))
            n_heads = int(base_spec.get("n_heads", 4))
            n_layers = int(base_spec.get("n_layers", 2))
            for heads in {max(1, n_heads // 2), n_heads, n_heads * 2}:
                if d_model % heads == 0:
                    variants.append(
                        {
                            "name": f"heads_{heads}",
                            "family": family,
                            **{**base_spec, "n_heads": heads},
                        }
                    )
            for depth in {max(1, n_layers - 1), n_layers, n_layers + 1}:
                variants.append({"name": f"depth_{depth}", "family": family, **{**base_spec, "n_layers": depth}})
            for activation in ("gelu", "relu", "silu"):
                variants.append(
                    {"name": f"act_{activation}", "family": family, **{**base_spec, "activation": activation}}
                )
            for norm in ("layernorm", "rmsnorm"):
                variants.append(
                    {"name": f"norm_{norm}", "family": family, **{**base_spec, "norm": norm}}
                )
        elif family == "mlp":
            hidden = list(base_spec.get("hidden_dims", [64]))
            variants.append({"name": "wider", "family": family, **{**base_spec, "hidden_dims": [h * 2 for h in hidden]}})
            variants.append({"name": "deeper", "family": family, **{**base_spec, "hidden_dims": hidden + hidden}})
        # dedupe by name
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for v in variants:
            if v["name"] in seen:
                continue
            seen.add(v["name"])
            unique.append(v)
        return unique

    def explore(
        self,
        *,
        family: str,
        base_spec: Mapping[str, Any],
        variants: Sequence[Mapping[str, Any]] | None = None,
        seed: int | str | None = 0,
        scope: TaskScope = TaskScope.MODEL_ARCHITECTURE_RESEARCH,
        batch: int = 2,
        seq: int = 8,
    ) -> dict[str, Any]:
        if scope not in (TaskScope.MODEL_ARCHITECTURE_RESEARCH, TaskScope.ARCHITECTURE_SYNTHESIS):
            raise InvalidOperationParameters(
                f"architecture research requires an architecture scope, got '{scope.value}'"
            )
        candidates_specs = list(variants or self.default_variants(family, base_spec))
        results: list[ArchitectureCandidate] = []
        for entry in candidates_specs:
            name = str(entry.get("name", "candidate"))
            spec = {k: v for k, v in entry.items() if k not in ("name", "family")}
            fam = str(entry.get("family", family))
            try:
                definition = self.synth.synthesize({"family": fam, **spec})
                package = definition.build_package(seed=_seed(seed), name=name)
                definition.graph.require_valid(package)
                inputs = definition.sample_inputs(batch=batch, seq=seq, seed=_seed(seed))
                timings: list[float] = []
                result = None
                for _ in range(2):
                    start = time.perf_counter()
                    result = execute(definition.graph, inputs, package)
                    timings.append(time.perf_counter() - start)
                assert result is not None
                results.append(
                    ArchitectureCandidate(
                        name=name,
                        spec={"family": fam, **spec},
                        ok=True,
                        num_nodes=len(definition.graph.nodes),
                        num_parameters=len(definition.parameter_shapes),
                        parameter_numel=int(sum(int(np.prod(s)) for s in definition.parameter_shapes.values())),
                        latency_ms=min(timings) * 1000.0,
                        output_shapes={k: list(v.shape) for k, v in result.outputs.items()},
                        definition=definition,
                    )
                )
            except Exception as exc:
                results.append(
                    ArchitectureCandidate(
                        name=name,
                        spec={"family": fam, **spec},
                        ok=False,
                        problems=[f"{type(exc).__name__}: {exc}"],
                    )
                )
        ok = [r for r in results if r.ok]
        pareto = self._pareto(ok)
        self.engine.audit.append(
            "architecture_research",
            {
                "family": family,
                "candidates": len(results),
                "valid": len(ok),
                "pareto": [c.name for c in pareto],
            },
        )
        return {
            "family": family,
            "base_spec": dict(base_spec),
            "candidates": [r.to_dict() for r in results],
            "pareto_front": [c.name for c in pareto],
            "valid": len(ok),
            "total": len(results),
            "measurement_notes": [
                "latency is the minimum of two CPU forward passes with random weights",
                "parameter_numel counts all bound parameters",
                "weights are structured random; outputs are not semantically meaningful",
            ],
        }

    def _pareto(self, candidates: Sequence[ArchitectureCandidate]) -> list[ArchitectureCandidate]:
        front: list[ArchitectureCandidate] = []
        for c in candidates:
            dominated = False
            for other in candidates:
                if other is c:
                    continue
                better_or_equal = (
                    other.parameter_numel <= c.parameter_numel
                    and (other.latency_ms or 0.0) <= (c.latency_ms or 0.0)
                )
                strictly_better = (
                    other.parameter_numel < c.parameter_numel
                    or (other.latency_ms or 0.0) < (c.latency_ms or 0.0)
                )
                if better_or_equal and strictly_better:
                    dominated = True
                    break
            if not dominated:
                front.append(c)
        return sorted(front, key=lambda c: c.parameter_numel)

    def graph_report(self, graph: Graph) -> dict[str, Any]:
        return {
            "summary": graph.summary(),
            "validation": graph.validate().to_dict(),
        }


def _seed(seed: int | str | None) -> int:
    from uzen.util.rng import parse_seed

    return parse_seed(seed)
