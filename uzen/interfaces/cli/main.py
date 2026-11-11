"""CLI for UZEN.

Terminal access to every capability family. The CLI is a thin, explicit layer over
the SDK/engines: no result is synthesised for display, and errors are printed with
their machine-readable code and exit with a non-zero status.

Examples::

    uzen ops list --family merging
    uzen policy show
    uzen families
    uzen inspect model.safetensors --full
    uzen import model.safetensors --family transformer_decoder
    uzen modify ARTIFACT_ID interpolate --other OTHER_ID --param t=0.5
    uzen generate --family mlp --spec '{"input_dim":8,"hidden_dims":[16],"output_dim":4}'
    uzen evaluate ARTIFACT_ID --family mlp --spec '{"input_dim":8,"hidden_dims":[16],"output_dim":4}'
    uzen align A B --method procrustes
    uzen research weights ARTIFACT_ID
    uzen research arch --family transformer_decoder --spec '{"d_model":16,"n_heads":4,"n_layers":1,"d_ff":32,"vocab_size":32}'
    uzen job submit request.json --run
    uzen serve --host 0.0.0.0 --port 8080
    uzen worker run --root ./worker-workspace
    uzen mcp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from uzen.errors import UzenError


def _json_arg(value: str | None, *, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON argument: {exc}")


def _kv_params(pairs: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--param expects key=value, got '{pair}'")
        key, _, raw = pair.partition("=")
        out[key.strip()] = _json_arg(raw.strip(), default=raw.strip())
    return out


def _emit(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _client(args: argparse.Namespace):
    from uzen.interfaces.python_sdk.client import Uzen

    if getattr(args, "url", None):
        return Uzen(base_url=args.url, token=getattr(args, "token", None))
    return Uzen(root=getattr(args, "root", "./uzen-workspace"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uzen", description="UZEN headless neural engineering substrate")
    parser.add_argument("--root", default="./uzen-workspace", help="local workspace root")
    parser.add_argument("--url", default=None, help="remote control plane base URL")
    parser.add_argument("--token", default=None, help="bearer token for remote access")
    sub = parser.add_subparsers(dest="command", required=True)

    # catalogue ------------------------------------------------------- #
    ops = sub.add_parser("ops", help="operation catalogue")
    ops_sub = ops.add_subparsers(dest="ops_command", required=True)
    ops_list = ops_sub.add_parser("list")
    ops_list.add_argument("--family")
    ops_list.add_argument("--executable-only", action="store_true")
    ops_show = ops_sub.add_parser("show")
    ops_show.add_argument("code")

    pol = sub.add_parser("policy", help="STRICT_PLUS policy")
    pol_sub = pol.add_subparsers(dest="policy_command", required=True)
    pol_sub.add_parser("show")

    sub.add_parser("families", help="architecture families")

    # inspection ------------------------------------------------------ #
    inspect = sub.add_parser("inspect", help="inspect a checkpoint safely")
    inspect.add_argument("path")
    inspect.add_argument("--header-only", action="store_true")

    imp = sub.add_parser("import", help="import a checkpoint into the artifact store")
    imp.add_argument("path")
    imp.add_argument("--name")
    imp.add_argument("--family")

    analyze = sub.add_parser("analyze", help="analyze an artifact")
    analyze.add_argument("artifact_id")

    # engineering ----------------------------------------------------- #
    modify = sub.add_parser("modify", help="apply a registered modification to an artifact")
    modify.add_argument("artifact_id")
    modify.add_argument("method")
    modify.add_argument("--other", action="append", default=[], help="additional artifact id")
    modify.add_argument("--param", action="append", default=[])
    modify.add_argument("--seed", default="0")

    gen = sub.add_parser("generate", help="generate weights")
    gen.add_argument("--family")
    gen.add_argument("--spec")
    gen.add_argument("--template")
    gen.add_argument("--method", default="orthogonal")
    gen.add_argument("--param", action="append", default=[])
    gen.add_argument("--seed", default="0")

    ev = sub.add_parser("evaluate", help="verify an artifact")
    ev.add_argument("artifact_id")
    ev.add_argument("--reference")
    ev.add_argument("--family")
    ev.add_argument("--spec")

    align = sub.add_parser("align", help="align one artifact to another")
    align.add_argument("source")
    align.add_argument("target")
    align.add_argument("--method", default="auto")
    align.add_argument("--threshold", type=float, default=0.5)

    # research -------------------------------------------------------- #
    research = sub.add_parser("research", help="architecture research")
    research_sub = research.add_subparsers(dest="research_command", required=True)
    rw = research_sub.add_parser("weights")
    rw.add_argument("artifact_id")
    ra = research_sub.add_parser("arch")
    ra.add_argument("--family", default="transformer_decoder")
    ra.add_argument("--spec", required=True)
    ra.add_argument("--variants")

    # artifacts ------------------------------------------------------- #
    art = sub.add_parser("artifacts", help="artifact store")
    art_sub = art.add_subparsers(dest="artifacts_command", required=True)
    art_sub.add_parser("list")
    art_show = art_sub.add_parser("show")
    art_show.add_argument("artifact_id")
    art_verify = art_sub.add_parser("verify")
    art_verify.add_argument("artifact_id")
    art_lineage = art_sub.add_parser("lineage")
    art_lineage.add_argument("artifact_id")
    art_export = art_sub.add_parser("export")
    art_export.add_argument("artifact_id")
    art_export.add_argument("destination")
    art_export.add_argument("--format", default="safetensors")

    # jobs ------------------------------------------------------------ #
    job = sub.add_parser("job", help="job lifecycle")
    job_sub = job.add_subparsers(dest="job_command", required=True)
    job_submit = job_sub.add_parser("submit")
    job_submit.add_argument("request_file", nargs="?", help="JSON request file ('-' for stdin)")
    job_submit.add_argument("--json", help="inline JSON request")
    job_submit.add_argument("--run", action="store_true")
    job_submit.add_argument("--idempotency-key")
    job_get = job_sub.add_parser("get")
    job_get.add_argument("job_id")
    job_list = job_sub.add_parser("list")
    job_list.add_argument("--state")
    job_list.add_argument("--limit", type=int, default=20)
    job_run = job_sub.add_parser("run")
    job_run.add_argument("job_id")
    job_cancel = job_sub.add_parser("cancel")
    job_cancel.add_argument("job_id")
    job_events = job_sub.add_parser("events")
    job_events.add_argument("job_id")
    job_events.add_argument("--after", type=int, default=0)

    # services -------------------------------------------------------- #
    serve = sub.add_parser("serve", help="run the REST control plane")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--reload", action="store_true")

    worker = sub.add_parser("worker", help="worker daemon")
    worker_sub = worker.add_subparsers(dest="worker_command", required=True)
    worker_run = worker_sub.add_parser("run")
    worker_run.add_argument("--root", default="./uzen-worker")
    worker_run.add_argument("--url", help="control plane URL (remote mode)")
    worker_run.add_argument("--secret")
    worker_run.add_argument("--worker-id", default=None)
    worker_run.add_argument("--once", action="store_true")
    worker_run.add_argument("--poll", type=float, default=1.0)

    mcp = sub.add_parser("mcp", help="run the MCP stdio server")
    mcp.add_argument("--root", default="./uzen-workspace")

    sub.add_parser("sdk", help="SDK introspection")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except UzenError as exc:
        print(json.dumps({"error": exc.to_dict()}, indent=2), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(json.dumps({"error": {"code": "file.not_found", "message": str(exc)}}, indent=2), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _dispatch(args: argparse.Namespace) -> int:
    cmd = args.command

    if cmd == "ops":
        if args.ops_command == "list":
            client = _client(args)
            items = client.operations()
            if args.family:
                items = [i for i in items if i["family"] == args.family]
            if args.executable_only:
                items = [i for i in items if i["maturity"] in ("implemented", "prototype")]
            _emit({"count": len(items), "operations": items})
            return 0
        client = _client(args)
        matches = [i for i in client.operations() if i["code"] == args.code]
        if not matches:
            raise SystemExit(f"unknown operation '{args.code}'")
        _emit(matches[0])
        return 0

    if cmd == "policy":
        _emit(_client(args).policy())
        return 0

    if cmd == "families":
        _emit({"families": _client(args).families()})
        return 0

    if cmd == "inspect":
        from uzen.core.wie.inspector import WeightIntelligenceEngine

        report = WeightIntelligenceEngine().inspect_file(args.path, full=not args.header_only)
        _emit(report.to_dict())
        return 0

    if cmd == "import":
        _emit(_client(args).import_weights(args.path, name=args.name, family=args.family))
        return 0

    if cmd == "analyze":
        _emit(_client(args).analyze(args.artifact_id))
        return 0

    if cmd == "modify":
        _emit(
            _client(args).modify(
                args.artifact_id,
                args.method,
                params=_kv_params(args.param),
                others=args.other,
                seed=args.seed,
            )
        )
        return 0

    if cmd == "generate":
        _emit(
            _client(args).generate(
                family=args.family,
                spec=_json_arg(args.spec),
                template_artifact_id=args.template,
                method=args.method,
                params=_kv_params(args.param),
                seed=args.seed,
            )
        )
        return 0

    if cmd == "evaluate":
        _emit(
            _client(args).evaluate(
                args.artifact_id,
                reference_artifact_id=args.reference,
                family=args.family,
                spec=_json_arg(args.spec),
            )
        )
        return 0

    if cmd == "align":
        _emit(
            _client(args).modify(
                args.source,
                "align",
                params={"threshold": args.threshold, "method": args.method},
                others=[args.target],
            )
        )
        return 0

    if cmd == "research":
        if args.research_command == "weights":
            _emit(
                _client(args).run_local(
                    {
                        "task_scope": "weight_architecture_research",
                        "objective": "compare weight representations",
                        "inputs": [{"name": "primary", "artifact_id": args.artifact_id}],
                    }
                )
            )
            return 0
        _emit(
            _client(args).run_local(
                {
                    "task_scope": "model_architecture_research",
                    "objective": "research architecture variants",
                    "options": {
                        "research_family": args.family,
                        "base_spec": _json_arg(args.spec),
                        "variants": _json_arg(args.variants),
                    },
                }
            )
        )
        return 0

    if cmd == "artifacts":
        client = _client(args)
        if args.artifacts_command == "list":
            _emit({"artifacts": client.service.artifacts()})
        elif args.artifacts_command == "show":
            _emit(client.artifact(args.artifact_id))
        elif args.artifacts_command == "verify":
            _emit(client.verify_artifact(args.artifact_id))
        elif args.artifacts_command == "lineage":
            _emit(client.lineage(args.artifact_id))
        else:
            path = client.service.store.export(args.artifact_id, args.destination, fmt=args.format)
            _emit({"exported": str(path), "format": args.format})
        return 0

    if cmd == "job":
        client = _client(args)
        if args.job_command == "submit":
            if args.json:
                request = _json_arg(args.json)
            elif args.request_file:
                raw = sys.stdin.read() if args.request_file == "-" else Path(args.request_file).read_text()
                request = json.loads(raw)
            else:
                raise SystemExit("provide a request file, '-' for stdin, or --json")
            job = client.submit_job(request, idempotency_key=args.idempotency_key)
            if args.run:
                _emit(client.run_job(job["id"]))
            else:
                _emit(job)
            return 0
        if args.job_command == "get":
            _emit(client.job(args.job_id))
            return 0
        if args.job_command == "list":
            _emit({"jobs": client.jobs(state=args.state, limit=args.limit)})
            return 0
        if args.job_command == "run":
            _emit(client.run_job(args.job_id))
            return 0
        if args.job_command == "cancel":
            _emit(client.cancel_job(args.job_id))
            return 0
        _emit({"events": client.job_events(args.job_id, after=args.after)})
        return 0

    if cmd == "serve":
        from uzen.interfaces.api.app import create_app

        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("uvicorn is required for `uzen serve`: pip install 'uzen[api]'") from exc
        app = create_app(root=args.root)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    if cmd == "worker":
        from uzen.workers.daemon import WorkerDaemon

        daemon = WorkerDaemon(
            root=args.root,
            worker_id=args.worker_id,
            control_plane=args.url,
            secret=args.secret,
        )
        if args.once:
            _emit(daemon.run_once())
            return 0
        daemon.run_forever(poll_interval=args.poll)
        return 0

    if cmd == "mcp":
        from uzen.interfaces.mcp.server import main as mcp_main

        return mcp_main(["--root", args.root])

    if cmd == "sdk":
        _emit(
            {
                "version": "0.1.0",
                "local": "Uzen(root=...)",
                "remote": "Uzen(base_url=...)",
                "docs": "docs/API.md",
            }
        )
        return 0

    raise SystemExit(f"unknown command '{cmd}'")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
