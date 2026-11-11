"""End-to-end UZEN example: generate -> modify -> verify -> export -> align.

Run with::

    python examples/quickstart.py ./uzen-demo

Everything here executes real tensor math through the operation registry; no value is
faked. Requires only numpy + pydantic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from uzen import Uzen
from uzen.core.npa.aligner import ParameterAligner
from uzen.core.nws.families import build_architecture
from uzen.core.vel.verify import verify


def main(root: str = "./uzen-demo") -> int:
    client = Uzen(root=root)

    spec = dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=64, max_seq_len=16)
    print("== 1. generate a tiny transformer decoder ==")
    generated = client.generate(family="transformer_decoder", spec=spec, seed=0)
    print(
        json.dumps(
            {
                "artifact": generated["artifact_id"],
                "verification_ok": generated["verification"]["ok"],
                "functional_ok": generated["verification"]["functional"]["ok"],
                "num_tensors": generated["num_tensors"],
                "origins": sorted(set(generated["origins"].values())),
            },
            indent=2,
        )
    )

    print("== 2. weight-only modification: SVD truncation ==")
    compressed = client.modify(generated["artifact_id"], "svd", params={"rank": 8})
    print(
        json.dumps(
            {
                "artifact": compressed["artifact_id"],
                "architecture_preserved": compressed["architecture_preserved"],
                "details": compressed["operation"]["details"],
            },
            indent=2,
            default=str,
        )[:900]
    )

    print("== 3. evaluate against the source architecture ==")
    report = client.evaluate(
        compressed["artifact_id"],
        reference_artifact_id=generated["artifact_id"],
        family="transformer_decoder",
        spec=spec,
    )
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "architecture_unchanged": report["structural"]["architecture_unchanged"],
                "worst_relative_l2": report["regression"]["worst_relative_l2_diff"],
                "latency_ms": report["performance"].get("mean_ms"),
            },
            indent=2,
            default=str,
        )
    )

    print("== 4. provenance + integrity ==")
    print(json.dumps(client.lineage(compressed["artifact_id"]), indent=2, default=str)[:800])
    print("integrity:", client.verify_artifact(compressed["artifact_id"])["ok"])

    print("== 5. export ==")
    path = client.service.store.export(
        compressed["artifact_id"], Path(root) / "exports" / "compressed.safetensors"
    )
    print("exported:", path)

    print("== 6. alignment and regression between two related parameter sets ==")
    definition = build_architecture("mlp", dict(input_dim=8, hidden_dims=[16], output_dim=4))
    base = definition.build_package(seed=1)
    variant = base.with_tensor("head.weight", base.tensors["head.weight"] * 0.9)
    alignment = ParameterAligner(method="procrustes", threshold=0.5).align(base, variant)
    print("alignment ok:", alignment.ok)

    inputs = definition.sample_inputs(batch=2, seed=0)
    result = verify(variant, reference=base, graph=definition.graph, inputs=inputs)
    print(
        "verification of variant:",
        result.ok,
        "| worst relative L2:",
        result.regression.get("worst_relative_l2_diff"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "./uzen-demo"))
