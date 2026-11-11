# UZEN — Universal Neural Weight Engineering & Architecture Research System

UZEN is a **headless computational substrate for engineering neural networks** without
training. It treats weights and architectures as programmable mathematical objects:
inspect, understand, research, design, generate, transform, compose, reconstruct,
evaluate, verify, export.

It is not a chatbot, an agent wrapper, a training platform, a model host, or a SaaS
frontend. There is no mandatory UI. Every capability is reachable through the Python
SDK, CLI, REST API and MCP server.

```
STRICT_PLUS: no training · no fine-tuning · no backpropagation · no gradient descent
```

---

## Status — honest capability report

| Area | State |
| --- | --- |
| STRICT_PLUS policy engine + hash-chained audit log | **implemented** |
| NWS tensor substrate, parameter roles, content-addressed artifact store | **implemented** |
| Safetensors codec (from scratch, header validation, BJNT-safe BF16) | **implemented** |
| NIR typed dataflow graph (symbolic dims, control-flow `select`, MoE/SSM/GNN/sampling kernels) | **implemented** |
| Weight operation registry (45 registered operations) | **implemented** |
| Arithmetic, merging (incl. TIES/DARE), decomposition (SVD/orthogonal/sparsify) | **implemented** |
| NPA alignment: identity, Procrustes, linear map, permutation (Hungarian), basis, activation/CKA | **implemented** |
| NASE synthesis: 12 families (transformer/MLP/CNN/RNN/autoencoder/MoE/SSM/GNN/VAE/diffusion/multimodal/hybrid), compose, substitute, bypass, truncate | **implemented** |
| WComp compiler + WPL language (real lexer, recursive-descent parser, typed pipeline) | **implemented** |
| VEL verification: structural, numerical, functional, regression, performance, provenance | **implemented** |
| ANE planning, candidate generation, 6 scalar derivative-free algorithms + NSGA-II Pareto search, diagnosis | **implemented** |
| WGE / WME / WAR / MAR engines | **implemented** (built on NWPC/NASE) |
| WIE inspection, statistics, role inference, family hypothesis | **implemented** |
| REST API (FastAPI), Python SDK, CLI, MCP stdio server, worker daemon | **implemented** |
| PyTorch runtime adapter (safe state-dict IO, inference-mode forward) | **implemented and tested with CPU-only torch 2.15; CUDA is never claimed** |
| Torch-CPU NIR execution (all 39 operators, parity-tested against NumPy) | **implemented** |
| GGUF v3 adapter + GGML type registry (F32/F16, Q4/Q5/Q8_0/Q4_K/Q6_K decode, Q8_0 encode, payload-free inspection) | **implemented** |
| ONNX interchange (export/import of the supported NIR subset) | **implemented** (requires the `onnx` extra; validated with `onnx.checker`) |
| Sparse-dispatch MoE (only routed tokens are computed per expert) | **implemented** |
| Prometheus `/metrics` + structured JSON logging (no extra dependencies) | **implemented** |
| Embedded ONNX runtime / GGUF quantized tensor types | **planned** (typed refusal, no simulation) |
| Rust / C++/CUDA kernels, native compiler | **planned** |
| Distributed object-store artifact transport | **planned** (shared-filesystem adapter implemented) |
| LLM-assisted planning | **implemented when an OpenAI-compatible endpoint is configured** |

Anything not implemented reports `planned`/`unsupported` and raises a typed error.
Random initialization is always labelled `randomly_initialized`. No operation claims
CUDA acceleration that does not exist.

---

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .            # core: numpy + pydantic
.venv/bin/pip install -e '.[api,dev]' # REST API + test tooling
.venv/bin/pip install -e '.[onnx]'    # optional: ONNX interchange
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu  # optional CPU-only adapter
```

The engine itself needs only `numpy` and `pydantic`. FastAPI/uvicorn are optional
extras used solely by the control-plane service.

## Quick start (SDK)

```python
from uzen import Uzen

client = Uzen(root="./uzen-workspace")

# 1. Build a real architecture (tiny transformer decoder) and its parameters.
model = client.generate(
    family="transformer_decoder",
    spec=dict(d_model=64, n_heads=4, n_layers=2, d_ff=128, vocab_size=128, max_seq_len=32),
    seed=0,
)
print(model["artifact_id"], model["verification"]["functional"]["ok"])

# 2. Weight-only modification: linear interpolation between two artifacts (t=0.5).
merged = client.modify(model["artifact_id"], "interpolate", others=[model["artifact_id"]], params={"t": 0.5})
print(merged["output_artifact_id"])

# 3. Verify provenance + integrity.
print(client.verify_artifact(merged["output_artifact_id"])["ok"])
```

## CLI

```bash
uzen families
uzen ops list --executable-only
uzen policy show
uzen inspect model.safetensors
uzen import model.safetensors --family transformer_decoder
uzen modify <artifact_id> interpolate --other <artifact_id> --param t=0.5
uzen evaluate <artifact_id> --family mlp --spec '{"input_dim":8,"hidden_dims":[16],"output_dim":4}'
uzen serve --host 0.0.0.0 --port 8080
uzen mcp --root ./uzen-workspace
uzen worker run --root ./uzen-worker
```

## REST control plane

`create_app()` exposes a versioned OpenAPI 3.1 surface (`/docs` for the schema):
jobs, models, weights, artifacts, architectures, operations, workers, policy.
Set `UZEN_API_TOKEN` to require bearer auth; worker routes additionally verify each
worker's registered secret hash.

```bash
curl -s localhost:8080/v1/operations | jq '.count'
curl -s -X POST localhost:8080/v1/jobs -H 'content-type: application/json' \
  -d '{"task_scope":"weight_modification","objective":"scale a model",
       "inputs":[{"name":"base","artifact_id":"<id>"}],
       "operations":[{"op":"w.arithmetic.scale","params":{"factor":0.5}}]}'
```

## MCP

`uzen mcp` speaks JSON-RPC 2.0 over stdio (newline or `Content-Length` framing) and
exposes deliberate tools — `uzen.inspect_model`, `uzen.modify_weights`,
`uzen.generate_weights`, `uzen.evaluate_candidate`, `uzen.inspect_provenance`, and
others. Each tool enforces task scope and STRICT_PLUS; an AI client cannot widen scope
or request a prohibited effect.

## The two engineering domains

Weights and architectures are **separate** engineering domains:

* **Weight engineering** (`weight_generation`, `weight_modification`,
  `weight_reconstruction`, `weight_optimization`, `weight_architecture_research`)
  changes parameters only. The structure digest of the output equals the source's.
* **Model/architecture engineering** (`model_creation`, `model_modification`,
  `model_reconstruction`, `model_architecture_research`, `architecture_synthesis`)
  changes graphs. The policy engine refuses architecture effects inside a weight-scope
  job and parameter effects inside an architecture-scope job.

A job's authorized scope is fixed at submission and preserved through planning,
registry dispatch, execution and artifact validation.

## Repository layout

```text
uzen/
├── core/
│   ├── nws/      # tensors, roles, manifests, families, artifact store
│   ├── nir/      # typed graph, operator registry/kernels, executor
│   ├── wpl/      # lexer + recursive-descent parser
│   ├── wie/      # weight intelligence / inspection
│   ├── nwpc/     # operation registry, operation context
│   ├── npa/      # neural parameter alignment
│   ├── nase/     # architecture synthesis
│   ├── wcomp/    # compiler, pipeline, execution plan
│   ├── vel/      # verification & evaluation lab
│   └── ane/      # autonomous neural engineer, search, planning, LLM bridge
├── engines/      # wge, wme, war, mar
├── operations/   # arithmetic, merging, decomposition, structural, generation,
│                 # reconstruction, alignment, search
├── runtimes/     # numpy reference runtime, PyTorch adapter
├── workers/      # protocol messages, worker daemon
├── interfaces/   # REST API, MCP server, CLI, Python SDK
├── infrastructure/ # storage (safetensors, SQLite), security, provenance
├── policies/     # strict_plus
└── jobs/         # job service (submit/run/cancel/events)
```

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module contracts and data flow.
* [docs/STRICT_PLUS.md](docs/STRICT_PLUS.md) — policy semantics and enforcement points.
* [docs/CAPABILITY_MATRIX.md](docs/CAPABILITY_MATRIX.md) — every operation, maturity and assumptions.
* [docs/API.md](docs/API.md) — REST/SDK/CLI/MCP contracts and error codes.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite exercises the 15 mandatory acceptance tests against real implementations:
real tensor arithmetic, real SVD, real alignment refusals, real policy rejection, real
job retries, real worker authentication and real graph binding validation. There are
no mocked computations.

## Explicit non-goals

* Turning UZEN into a web application, dashboard or SaaS.
* TypeScript as the neural computation language.
* Claiming a module is implemented because a route or database table exists.
* Producing "improved" models from random weights without a mathematical justification.
* Overwriting source weights, or emitting artifacts without provenance and validation.
