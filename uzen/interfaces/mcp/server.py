"""MCP server for UZEN.

Exposes deliberate, high-level neural-engineering tools over the Model Context
Protocol (JSON-RPC 2.0 on stdio). Each tool enforces task scope and STRICT_PLUS
independently of the calling AI client: an agent cannot talk UZEN into training,
fine-tuning or gradient updates, and cannot silently widen a job's scope.

Framing: both newline-delimited JSON and ``Content-Length``-prefixed messages are
accepted on stdin, so the server works with either MCP transport convention.

Run::

    uzen mcp --root ./uzen-workspace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from uzen.errors import UzenError
from uzen.interfaces.python_sdk.client import Uzen

SERVER_INFO = {"name": "uzen", "version": "0.1.0"}
PROTOCOL_VERSION = "2024-11-05"


def _text(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}]}


class MCPServer:
    def __init__(self, *, root: str | Path = "./uzen-workspace") -> None:
        self.client = Uzen(root=root)
        self.tools: dict[str, tuple[dict[str, Any], Callable[[Mapping[str, Any]], Any]]] = {
            "uzen.inspect_model": (
                {
                    "description": "Inspect a checkpoint or artifact: tensors, dtypes, roles, statistics, family hypothesis.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "artifact_id": {"type": "string"},
                        },
                    },
                },
                self._inspect_model,
            ),
            "uzen.analyze_weights": (
                {
                    "description": "Statistical analysis of an artifact's tensors and parameter roles.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"artifact_id": {"type": "string"}},
                        "required": ["artifact_id"],
                    },
                },
                self._analyze_weights,
            ),
            "uzen.plan_engineering": (
                {
                    "description": "Produce an executable engineering plan for an objective without executing it.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "task_scope": {"type": "string"},
                            "objective": {"type": "string"},
                            "inputs": {"type": "array"},
                            "evaluation_plan": {"type": "object"},
                            "budget": {"type": "object"},
                        },
                        "required": ["task_scope", "objective"],
                    },
                },
                self._plan,
            ),
            "uzen.generate_weights": (
                {
                    "description": "Generate parameters from an architecture family or a template artifact. Output is labelled as non-learned.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "family": {"type": "string"},
                            "spec": {"type": "object"},
                            "template_artifact_id": {"type": "string"},
                            "method": {"type": "string"},
                            "seed": {},
                        },
                    },
                },
                self._generate,
            ),
            "uzen.modify_weights": (
                {
                    "description": "Apply a registered weight modification (interpolate, merge, SVD, align, sparsify...).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "artifact_id": {"type": "string"},
                            "method": {"type": "string"},
                            "params": {"type": "object"},
                            "others": {"type": "array"},
                            "seed": {},
                        },
                        "required": ["artifact_id", "method"],
                    },
                },
                self._modify,
            ),
            "uzen.reconstruct_weights": (
                {
                    "description": "Reconstruct a complete parameter set from partial components.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "inputs": {"type": "array"},
                            "operations": {"type": "array"},
                            "seed": {},
                        },
                        "required": ["inputs"],
                    },
                },
                self._reconstruct,
            ),
            "uzen.research_weight_architecture": (
                {
                    "description": "Measure representation trade-offs (low-rank, sparsity, basis changes).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"artifact_id": {"type": "string"}, "experiments": {"type": "array"}},
                        "required": ["artifact_id"],
                    },
                },
                self._research_weights,
            ),
            "uzen.research_model_architecture": (
                {
                    "description": "Explore architecture variants and measure cost/shape properties.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "family": {"type": "string"},
                            "base_spec": {"type": "object"},
                            "variants": {"type": "array"},
                        },
                        "required": ["family", "base_spec"],
                    },
                },
                self._research_arch,
            ),
            "uzen.modify_architecture": (
                {
                    "description": "Graph surgery: bypass a node or truncate layers of a stored architecture.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "artifact_id": {"type": "string"},
                            "operation": {"type": "string"},
                            "node_id": {"type": "string"},
                            "keep_layers": {"type": "integer"},
                        },
                        "required": ["artifact_id", "operation"],
                    },
                },
                self._modify_architecture,
            ),
            "uzen.create_model": (
                {
                    "description": "Create a model architecture with initial parameters and validate a forward pass.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"family": {"type": "string"}, "spec": {"type": "object"}, "seed": {}},
                        "required": ["family"],
                    },
                },
                self._create_model,
            ),
            "uzen.evaluate_candidate": (
                {
                    "description": "Verify an artifact: structure, numerical validity, forward behaviour, performance.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "artifact_id": {"type": "string"},
                            "reference_artifact_id": {"type": "string"},
                            "family": {"type": "string"},
                            "spec": {"type": "object"},
                        },
                        "required": ["artifact_id"],
                    },
                },
                self._evaluate,
            ),
            "uzen.inspect_provenance": (
                {
                    "description": "Artifact manifest, lineage chain and integrity verification.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"artifact_id": {"type": "string"}},
                        "required": ["artifact_id"],
                    },
                },
                self._provenance,
            ),
            "uzen.get_job_status": (
                {
                    "description": "Job state, events and results.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"job_id": {"type": "string"}},
                        "required": ["job_id"],
                    },
                },
                self._job_status,
            ),
        }

    # ------------------------------------------------------------------ #
    # JSON-RPC
    # ------------------------------------------------------------------ #

    def handle(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}
        if method == "initialize":
            return _result(
                msg_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "serverInfo": SERVER_INFO,
                    "capabilities": {"tools": {}, "prompts": {}},
                    "instructions": (
                        "UZEN is a headless neural weight/architecture engineering substrate. "
                        "STRICT_PLUS: no training, fine-tuning, backpropagation or gradient "
                        "updates. Tools are scope-checked; unregistered operations are refused."
                    ),
                },
            )
        if method in ("notifications/initialized", "initialized"):
            return None
        if method == "ping":
            return _result(msg_id, {})
        if method == "tools/list":
            return _result(
                msg_id,
                {
                    "tools": [
                        {"name": name, "description": schema["description"], "inputSchema": schema["inputSchema"]}
                        for name, (schema, _fn) in sorted(self.tools.items())
                    ]
                },
            )
        if method == "tools/call":
            name = str(params.get("name", ""))
            arguments = params.get("arguments") or {}
            entry = self.tools.get(name)
            if entry is None:
                return _error(msg_id, -32602, f"unknown tool '{name}'")
            schema, fn = entry
            try:
                payload = fn(arguments)
            except UzenError as exc:
                return _result(msg_id, {"isError": True, "content": [{"type": "text", "text": json.dumps(exc.to_dict())}]})
            except Exception as exc:  # reported, not hidden
                return _result(
                    msg_id,
                    {
                        "isError": True,
                        "content": [
                            {"type": "text", "text": json.dumps({"code": "tool.failed", "message": str(exc), "tool": name})}
                        ],
                    },
                )
            return _result(msg_id, _text(payload))
        if method in ("resources/list",):
            return _result(msg_id, {"resources": []})
        if method in ("prompts/list",):
            return _result(msg_id, {"prompts": []})
        return _error(msg_id, -32601, f"method not found: {method}")

    # ------------------------------------------------------------------ #
    # tool implementations
    # ------------------------------------------------------------------ #

    def _inspect_model(self, args: Mapping[str, Any]) -> Any:
        if args.get("artifact_id"):
            return self.client.analyze(str(args["artifact_id"]))
        path = args.get("path")
        if not path:
            raise UzenError("provide path or artifact_id")
        return self.client.inspect(str(path))

    def _analyze_weights(self, args: Mapping[str, Any]) -> Any:
        return self.client.analyze(str(args["artifact_id"]))

    def _plan(self, args: Mapping[str, Any]) -> Any:
        from uzen.core.ane.planner import JobRequest

        request = JobRequest.from_dict(dict(args))
        inputs = []
        for entry in request.inputs:
            if "name" in entry:
                inputs.append(str(entry["name"]))
            elif "artifact_id" in entry:
                inputs.append("input")
        plan = self.client.service.ane.plan(request, input_names=inputs or ["input"])
        return {"ok": True, "plan": plan.to_dict()}

    def _generate(self, args: Mapping[str, Any]) -> Any:
        return self.client.generate(
            family=args.get("family"),
            spec=args.get("spec"),
            template_artifact_id=args.get("template_artifact_id"),
            method=str(args.get("method", "orthogonal")),
            seed=args.get("seed", 0),
        )

    def _modify(self, args: Mapping[str, Any]) -> Any:
        return self.client.modify(
            str(args["artifact_id"]),
            str(args["method"]),
            params=args.get("params") or {},
            others=args.get("others") or [],
            seed=args.get("seed", 0),
        )

    def _reconstruct(self, args: Mapping[str, Any]) -> Any:
        return self.client.run_local(
            {
                "task_scope": "weight_reconstruction",
                "objective": "reconstruct weights from supplied components",
                "inputs": args["inputs"],
                "operations": args.get("operations") or [],
                "seed": args.get("seed", 0),
            }
        )

    def _research_weights(self, args: Mapping[str, Any]) -> Any:
        return self.client.run_local(
            {
                "task_scope": "weight_architecture_research",
                "objective": "compare weight representations",
                "inputs": [{"name": "primary", "artifact_id": args["artifact_id"]}],
                "operations": args.get("experiments") or [],
            }
        )

    def _research_arch(self, args: Mapping[str, Any]) -> Any:
        return self.client.run_local(
            {
                "task_scope": "model_architecture_research",
                "objective": "research architecture variants",
                "options": {
                    "research_family": args["family"],
                    "base_spec": args["base_spec"],
                    "variants": args.get("variants"),
                },
            }
        )

    def _modify_architecture(self, args: Mapping[str, Any]) -> Any:
        from uzen.core.nase.synth import ArchitectureSynthesizer
        from uzen.core.nir.graph import Graph
        from uzen.core.vel.verify import verify

        manifest = self.client.artifact(str(args["artifact_id"]))
        graph_doc = (manifest.get("metadata") or {}).get("graph")
        if not graph_doc:
            raise UzenError("artifact has no stored NIR graph; only synthesized models can be edited")
        package = self.client.package(str(args["artifact_id"]))
        synth = ArchitectureSynthesizer()
        from uzen.interfaces.python_sdk.client import Uzen as _U

        definition = synth.from_graph(
            Graph.from_dict(graph_doc),
            parameter_shapes={e["name"]: e["shape"] for e in manifest["tensors"]},
            family=manifest.get("family", "custom"),
        )
        operation = str(args.get("operation", "bypass"))
        if operation == "bypass":
            report = synth.bypass(definition, str(args["node_id"]))
        elif operation == "truncate":
            report = synth.truncate_layers(definition, int(args["keep_layers"]))
        else:
            raise UzenError(f"unsupported architecture operation '{operation}'")
        report.definition.graph.require_valid()
        inputs = {
            name: __import__("numpy").zeros(tuple(2 if not isinstance(d, int) else d for d in spec.shape), dtype=spec.dtype)
            for name, spec in report.definition.graph.inputs.items()
        }
        verification = verify(package, graph=report.definition.graph, inputs=inputs)
        return {
            "changes": report.changes,
            "invalidated_parameters": report.invalidated_parameters,
            "summary": report.definition.summary(),
            "graph": report.definition.graph.to_dict(),
            "verification": verification.to_dict(),
        }

    def _create_model(self, args: Mapping[str, Any]) -> Any:
        return self.client.generate(
            family=str(args["family"]), spec=args.get("spec") or {}, seed=args.get("seed", 0)
        )

    def _evaluate(self, args: Mapping[str, Any]) -> Any:
        return self.client.evaluate(
            str(args["artifact_id"]),
            reference_artifact_id=args.get("reference_artifact_id"),
            family=args.get("family"),
            spec=args.get("spec"),
        )

    def _provenance(self, args: Mapping[str, Any]) -> Any:
        artifact_id = str(args["artifact_id"])
        return {
            "manifest": self.client.artifact(artifact_id),
            "lineage": self.client.lineage(artifact_id),
            "verification": self.client.verify_artifact(artifact_id),
        }

    def _job_status(self, args: Mapping[str, Any]) -> Any:
        job_id = str(args["job_id"])
        return {"job": self.client.job(job_id), "events": self.client.job_events(job_id)}


def _result(msg_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


# --------------------------------------------------------------------------- #
# stdio transport
# --------------------------------------------------------------------------- #


def _read_message(stream: TextIO) -> dict[str, Any] | None:
    line = stream.readline()
    if not line:
        return None
    stripped = line.strip()
    if not stripped:
        return _read_message(stream)
    if stripped.lower().startswith("content-length:"):
        try:
            length = int(stripped.split(":", 1)[1].strip())
        except ValueError:
            return None
        while True:
            header = stream.readline()
            if not header or not header.strip():
                break
        body = stream.read(length)
        return json.loads(body) if body else None
    return json.loads(stripped)


def _write_message(stream: TextIO, payload: Mapping[str, Any], *, framed: bool) -> None:
    body = json.dumps(payload, ensure_ascii=False)
    if framed:
        stream.write(f"Content-Length: {len(body.encode('utf-8'))}\r\n\r\n{body}")
    else:
        stream.write(body + "\n")
    stream.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uzen-mcp")
    parser.add_argument("--root", default="./uzen-workspace")
    parser.add_argument("--framed", action="store_true", help="use Content-Length framing")
    args = parser.parse_args(argv)
    server = MCPServer(root=args.root)
    stdin, stdout = sys.stdin, sys.stdout
    while True:
        try:
            message = _read_message(stdin)
        except json.JSONDecodeError as exc:
            _write_message(stdout, _error(None, -32700, f"parse error: {exc}"), framed=args.framed)
            continue
        if message is None:
            return 0
        response = server.handle(message)
        if response is not None:
            _write_message(stdout, response, framed=args.framed)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
