# Architecture

UZEN is a layered computational substrate. Each layer has a narrow contract; the only
path that mutates parameters runs through the operation registry, and every produced
artifact carries provenance and validation evidence.

```text
            interfaces            REST · MCP · CLI · Python SDK
                 │
                 ▼
              jobs/               control plane: lifecycle, idempotency, events, leases
                 │
                 ▼
        core/ane (planner)        objective → plan → candidates → search → diagnosis
                 │
   ┌─────────────┼──────────────┬──────────────┐
   ▼             ▼              ▼              ▼
core/wie      core/npa       core/nwpc      core/nase
inspection    alignment      registry       synthesis
   │             │              │              │
   └─────────────┴──────┬───────┴──────────────┘
                        ▼
                 core/nir (typed graph + kernels)
                        │
                        ▼
                 core/nws (tensors + store)
                        │
                        ▼
        policies/strict_plus + infrastructure (storage, provenance, security)
```

## NWS — Neural Weight Substrate

`WeightPackage` is an immutable-by-convention bundle of

* `tensors: dict[str, np.ndarray]` — real arrays, no lazy placeholders;
* `metas: dict[str, TensorMeta]` — shape, dtype, inferred **role**, **origin**
  (`inherited`, `extracted`, `transformed`, `analytically_derived`,
  `randomly_initialized`, `search_selected`), source artifact, applied operations;
* `family` + `family_spec` — the architecture this parameter set belongs to;
* `fingerprint()` — SHA-256 over canonical tensor bytes (value identity);
* `structure_digest()` — SHA-256 over names/shapes/dtypes (architecture identity).

Weight-only operations must preserve `structure_digest()`; this is asserted in tests.

`ArtifactStore` is content-addressed:

```text
<root>/artifacts/<aa>/<sha256>/{manifest.json, tensors.safetensors, lineage.json}
```

`artifact_id` is derived from schema version, tensor fingerprint, family, operation,
parents, operations and seeds — so retries are idempotent and re-running a job cannot
mutate an existing artifact. Publication is atomic (write temp dir, then rename), and
`verify()` re-hashes the payload against the manifest checksum.

## NIR — Neural Intermediate Representation

A typed dataflow graph, not a layer list:

* `Node(id, op, inputs: dict[port, Ref], params)`, `Ref("node:port")` / `Ref("input:name")`;
* symbolic dims (`"B"`, `"T"`) resolved per execution; concrete dims checked exactly;
* per-operator shape inference (`OperatorSpec.infer_types`) and kernels
  (`OperatorSpec.run`) in one registry, so a graph cannot validate but fail to run;
* cycle detection via Kahn topological sort, with an explicit cycle error;
* `Graph.validate(package)` checks declared inputs, parameter binding, shapes and
  dtype compatibility; `require_valid` raises `GraphValidationError`.

The interpreter (`executor.execute`) binds parameters **by tensor name**, resolves
symbols, runs kernels in topological order, and raises `ShapeMismatch` with node id and
input shapes on failure. `capture=` returns intermediate activations for VEL/NPA.

Families (`transformer_decoder`, `mlp`, `cnn`, `recurrent`, `autoencoder`) construct
graphs from specs, allocate parameter shapes, and initialize analytically where a
closed form exists (orthogonal rows, unit norms, zero bias) or label tensors
`randomly_initialized` otherwise.

## NWPC — operation registry

The single choke point for parameter change. Each `OperationSpec` declares:

* `code` (`w.<family>.<name>`), human title, effect (`param_transform`,
  `param_generate`, `param_select`, `arch_mutation`, `analysis`, ...);
* `maturity` (`implemented` / `prototype` / `planned` / `unsupported`);
* allowed `scopes` (task-scope matrix);
* typed `params` (`ParamSpec` with ranges/choices) — unknown parameters are rejected,
  not ignored;
* `assumptions` — the mathematical conditions under which the operation is defensible;
* `requires_alignment`, `mutates_architecture`, `min/max_inputs`.

`invoke(code, inputs, params, scope=..., seed=...)` performs: registration check →
maturity check → scope check → STRICT_PLUS authorization → parameter validation →
function execution → origin/provenance composition. Inputs are never mutated; the
output is a new `WeightPackage` with per-tensor origins and an `OperationRecord`
containing policy decisions, seed, details and parent artifact ids.

`invoke_search` runs derivative-free search (CMA-ES, evolution strategies, Bayesian
optimization, coordinate descent, simulated annealing, beam search) over
low-dimensional transformation parameters only — never over billions of individual
weights.

## NPA — alignment

Two models can share shapes yet live in different learned bases. `ParameterAligner`
tries `identity`, `procrustes` (closed-form SVD), `linear_map` (ridge least squares),
`permutation` (Hungarian assignment on a cosine cost matrix, plus attention-head
grouping), `basis` (whitening/colouring) and `activation` (forward-pass CKA + neuron
matching). `auto` evaluates candidates and selects the best-scoring defensible one.

If the best score is below the threshold, the report is `ok=False` and `require()`
raises `NoValidAlignment` (`npa.no_valid_alignment`). The report always states
`semantic_equivalence_claimed: false`: a numeric score is evidence, not proof.

## NASE — architecture synthesis

Build graphs from specs, compose definitions (`sequential`/`parallel`), substitute
operators, bypass nodes, truncate layers. Every transformation validates the resulting
graph and reports invalidated parameters, so callers know exactly which tensors must
be re-bound. `from_graph` rebuilds an editable definition from a stored NIR graph.

## WComp / WPL

WPL is a small declarative language with a real lexer and recursive-descent parser:

```text
program merge_and_scale
  input base: weights
  input other: weights
  param t: float = 0.5
  param factor: float = 1.0
  step blend = linear_interpolate(base, other, t=t)
  step out = scale(blend, factor=factor)
  output result = out
end
```

`compile_program` resolves short operation names against the registry, type-checks
declared parameters, and produces a `CompiledPipeline` whose `CompilationReport`
records per-stage effects, maturity, policy decisions and compatibility. Execution
returns a `PipelineResult` with the output artifact id, provenance and per-stage
records.

## VEL — verification

Independent sections, each reported separately (they are not interchangeable):

* **structural** — tensor validity, shape/dtype consistency, architecture unchanged;
* **numerical** — finite values, distribution statistics, outlier/anomaly flags;
* **functional** — real forward pass, output shapes, finiteness, optional comparison
  against a reference (relative L2, allclose);
* **regression** — worst relative L2 difference across outputs;
* **performance** — latency statistics over repeated runs, device, memory estimate;
* **provenance** — lineage parents and operation records;
* **policy** — STRICT_PLUS decisions attached to the producing operations.

## ANE — autonomous engineer

`plan()` interprets the request, applies the scope matrix, selects operations by
objective keywords and available operations, and emits typed `PlanStage`s with search
spaces. `execute()` runs each stage, generates candidates (grid/random low-dimensional
sampling), scores them with an `Objective` (fidelity, distance-to-target, sparsity,
activation agreement, size, custom callables), searches within the stage's search
space, keeps the best candidate per stage, verifies the final package, and returns a
trace plus diagnosis. When alignment fails it tries the next defensible strategy; when
none exists it reports `NO_VALID_ALIGNMENT` rather than inventing a result.

`LLMPlanner` optionally consults an OpenAI-compatible endpoint, but every proposed
stage is re-validated through the same registry/scope/policy path.

## Control and execution planes

`JobService` owns a SQLite job database (states, events, leases, idempotency keys,
artifact index) plus the artifact store and engines. States:
`CREATED → QUEUED → DISPATCHED → RUNNING → VALIDATING → SUCCEEDED|FAILED`, with
`CANCEL_REQUESTED → CANCELLED`. Job requests are persisted before execution, so a
retry re-reads the same authorized scope.

`WorkerDaemon` runs jobs either locally or against the REST control plane
(register → heartbeat → claim under lease → execute → report). Workers advertise
detected capabilities (CPU/memory/Python/torch, supported operation codes, supported
families). They never execute user-supplied scripts: only registered operations over
artifacts.

## Design boundaries

* Base44 (or any host) can provide control-plane metadata, orchestration and API
  exposure. It must never pretend to execute tensor math.
* Tensor computation happens in Python/NumPy (reference) or PyTorch (adapter).
* Rust/C++/CUDA are future native layers, not stub modules.
