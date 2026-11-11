# API Contracts

All interfaces share one engine. Responses are JSON; errors carry a machine-readable
`code` and HTTP status.

## Error shape

```json
{
  "detail": {
    "code": "policy.violation",
    "message": "STRICT_PLUS prohibits effect 'gradient_update'",
    "details": {"operation": "w.train.finetune"}
  }
}
```

| Code | HTTP | Meaning |
| --- | --- | --- |
| `policy.violation` | 403 | prohibited effect or unauthorized plan |
| `policy.scope_violation` | 403 | effect outside the authorized task scope |
| `registry.not_registered` | 400 | operation code unknown |
| `operation.unavailable` | 409 | operation planned/unsupported or wrong scope |
| `params.invalid` | 400 | missing/unknown/out-of-range parameter |
| `tensor.shape_mismatch` | 400 | incompatible shapes or missing parameters |
| `tensor.invalid` | 400 | non-finite or corrupt tensors |
| `npa.no_valid_alignment` | 422 | no defensible parameter correspondence |
| `graph.invalid` | 400 | invalid NIR graph or parameter bindings |
| `format.unsupported` | 415 | format not supported (e.g. pickle checkpoints) |
| `artifact.not_found` | 404 | unknown artifact id |
| `artifact.immutable` | 409 | attempted rewrite of an existing artifact |
| `job.state` | 409 | illegal job transition |
| `auth.invalid` / `auth.missing` | 401 | bad or missing bearer token |
| `worker.auth_failed` | 401 | invalid worker credentials |
| `runtime.unavailable` | 503 | optional backend missing (e.g. torch) |
| `budget.exhausted` | 429 | search budget exhausted |

## REST (v1)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | liveness, policy name |
| GET | `/v1/stats` | store/db stats + detected capabilities |
| GET | `/v1/operations` | catalogue (`family`, `executable_only`) |
| GET | `/v1/operations/{code}` | one operation spec |
| GET | `/v1/policies/strict-plus` | policy description, effects, scopes |
| GET | `/v1/architectures/families` | supported families + specs |
| POST | `/v1/architectures/synthesize` | build a model from a family spec |
| POST | `/v1/architectures/research` | compare architecture variants |
| POST | `/v1/models/import` | import a checkpoint into the store |
| GET | `/v1/models` | model registry |
| GET | `/v1/weights` | weight artifact registry |
| POST | `/v1/weights/analyze` | WIE inspection (`artifact_id` or `path`) |
| POST | `/v1/weights/generate` | weight generation |
| POST | `/v1/weights/modify` | registered modification |
| POST | `/v1/weights/reconstruct` | reconstruction job |
| POST | `/v1/weights/evaluate` | VEL verification |
| GET | `/v1/artifacts/{id}` | manifest |
| GET | `/v1/artifacts/{id}/verify` | integrity verification |
| GET | `/v1/artifacts/{id}/lineage` | lineage chain |
| GET | `/v1/artifacts/{id}/export` | export (safetensors download) |
| POST | `/v1/jobs` | submit (`Idempotency-Key` supported) |
| GET | `/v1/jobs` | list (`state`, `limit`) |
| GET | `/v1/jobs/{id}` | job detail |
| POST | `/v1/jobs/{id}/run` | run synchronously |
| POST | `/v1/jobs/{id}/cancel` | cancel |
| GET | `/v1/jobs/{id}/events` | progress events |
| GET | `/v1/workers` | worker registry |
| POST | `/v1/workers/register` | register, returns one-time secret |
| POST | `/v1/workers/heartbeat` | liveness |
| POST | `/v1/workers/claim` | claim a job under a lease |
| POST | `/v1/workers/renew` | renew lease |
| POST | `/v1/workers/report` | report result |

Auth: `Authorization: Bearer <UZEN_API_TOKEN>` when the variable is set. Worker routes
carry `worker_id` + `secret` in the body; the control plane compares
`SHA-256(secret)` against the stored hash in constant time.

## Job request

```json
{
  "task_scope": "weight_modification",
  "objective": "blend two compatible parameter sets",
  "inputs": [{"name": "base", "artifact_id": "..."}, {"name": "other", "path": "model.safetensors"}],
  "operations": [{"op": "w.arithmetic.linear_interpolate", "params": {"t": 0.4}}],
  "constraints": {"max_tensors": 512},
  "policy": "STRICT_PLUS",
  "evaluation_plan": {"metric": "distance_to_target", "target_artifact_id": "..."},
  "output_format": "safetensors",
  "seed": 0,
  "budget": {"candidates": 8, "evaluations": 16, "search_algorithm": "cma_es"}
}
```

Inputs accept `artifact_id`, `path` or `inline` arrays. The job result contains the
plan, per-stage operation records, policy decisions, candidates with metrics, the best
candidate's artifact id, the VEL report, trace, warnings and diagnosis.

## Python SDK

```python
from uzen import Uzen

local  = Uzen(root="./ws")
remote = Uzen(base_url="http://localhost:8080", token="...")

local.inspect(path)                     # WIE report (no pickle)
local.import_weights(path, family=...)  # artifact id
local.analyze(artifact_id)
local.generate(family=..., spec=..., seed=0)
local.modify(artifact_id, "svd_truncate", params={"rank": 32})
local.evaluate(artifact_id, family=..., spec=...)
local.submit_job(request); local.run_job(job_id); local.job_events(job_id)
local.run_local(request)                # submit + run, returns full result
local.verify_artifact(artifact_id); local.lineage(artifact_id)
local.operations(); local.policy(); local.families()
```

## CLI

`uzen --help` lists groups: `ops`, `policy`, `families`, `inspect`, `import`,
`analyze`, `modify`, `generate`, `evaluate`, `align`, `research`, `artifacts`, `job`,
`serve`, `worker`, `mcp`, `sdk`. Errors print `{"error": {...}}` to stderr and exit 2.

## MCP

Tools: `uzen.inspect_model`, `uzen.analyze_weights`, `uzen.plan_engineering`,
`uzen.generate_weights`, `uzen.modify_weights`, `uzen.reconstruct_weights`,
`uzen.research_weight_architecture`, `uzen.research_model_architecture`,
`uzen.modify_architecture`, `uzen.create_model`, `uzen.evaluate_candidate`,
`uzen.inspect_provenance`, `uzen.get_job_status`.

Transport: JSON-RPC 2.0 over stdio; `initialize`, `tools/list`, `tools/call`, `ping`.
Tool failures return `isError: true` with the typed error payload rather than a
fabricated success.

## Worker protocol (`uzen.worker/1`)

1. `register` → `{worker_id, secret}` (secret shown once, stored as a hash).
2. `heartbeat` with detected capabilities.
3. `claim` → job or `null`; leases expire (`lease_seconds`), `renew` extends.
4. Execute locally through `JobService` (registered operations only).
5. `report` with state, result/error, duration.

Artifact transport in this build: shared filesystem (`exchange_dir`) or inline
tensors. Object-store transport is planned.
