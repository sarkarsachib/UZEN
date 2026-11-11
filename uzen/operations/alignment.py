"""Alignment operations exposed through the registry.

Alignment is not a cosmetic step: combining or transferring parameters across
different learned bases is mathematically unjustified without it. These operations
make the alignment decision explicit, auditable, and refusable — when no defensible
correspondence exists they raise ``NoValidAlignment`` rather than producing a
plausible-looking result.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from uzen.core.npa.aligner import ParameterAligner, METHODS
from uzen.core.nwpc.registry import OperationSpec, ParamSpec, default_registry
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import NoValidAlignment
from uzen.operations._common import make_outcome
from uzen.policies.strict_plus.effects import EngineEffect, Origin, TaskScope

_SCOPES = frozenset(
    {
        TaskScope.WEIGHT_MODIFICATION,
        TaskScope.WEIGHT_RECONSTRUCTION,
        TaskScope.WEIGHT_GENERATION,
        TaskScope.WEIGHT_OPTIMIZATION,
    }
)


def _align_op(method: str):
    def impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
        source, target = inputs[0], inputs[1]
        aligner = ParameterAligner(
            method=method,
            threshold=float(p["threshold"]),
            allow_partial=bool(p.get("allow_partial", False)),
        )
        report = aligner.align(source, target)
        if not report.ok:
            raise NoValidAlignment(
                f"alignment method '{method}' could not establish a defensible correspondence: "
                + "; ".join(report.reasons[:3]),
                method=method,
                score=float(report.score),
                threshold=float(p["threshold"]),
                skipped=report.skipped,
            )
        aligned = aligner.apply(source, report, include_untransformed=True)
        out = {
            name: ctx.emit(aligned.tensors[name], name=name)
            for name in aligned.names()
        }
        details = {
            "op": f"w.align.{method}",
            "method": method,
            "score": float(report.score),
            "threshold": float(p["threshold"]),
            "aligned_tensors": len(report.tensors),
            "skipped": report.skipped,
            "candidates": report.candidates,
            "semantic_equivalence_claimed": False,
            "note": (
                "source parameters were transformed into the target basis; the score measures "
                "numerical correspondence under that transform, not functional equivalence"
            ),
        }
        return make_outcome(
            op=f"w.align.{method}",
            tensors=out,
            replaced=set(out),
            details=details,
            origin=Origin.TRANSFORMED,
            alignment=report.to_dict(),
        )

    return impl


def _register(spec: OperationSpec) -> None:
    default_registry().register(spec)


for _method in ("identity", "procrustes", "linear_map", "permutation", "basis", "auto"):
    _register(OperationSpec(
        code=f"w.align.{_method}",
        title=f"Parameter alignment ({_method})",
        family="alignment",
        effect=EngineEffect.PARAM_TRANSFORM,
        fn=_align_op(_method),
        description=f"Align source parameters to a target basis using '{_method}'.",
        scopes=_SCOPES,
        params={
            "threshold": ParamSpec("float", default=0.5, minimum=0.0, maximum=1.0),
            "allow_partial": ParamSpec("bool", default=False),
        },
        assumptions=(
            "tensor names and shapes correspond across the two parameter sets",
            "an orthogonal/linear reparameterization is sufficient to recover the correspondence",
            f"method '{_method}' is applicable only when its numerical assumptions hold",
        ),
        requires_alignment=False,
        min_inputs=2,
        max_inputs=2,
        origin=Origin.TRANSFORMED,
    ))
