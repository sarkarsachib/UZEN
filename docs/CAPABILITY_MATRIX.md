# Capability Matrix

Generated from the live operation registry (`GET /v1/operations`, `uzen ops list`).
Maturity levels: `implemented` (executable and tested), `prototype` (executable with
documented limits), `planned` (designed, not executable), `unsupported`.

No operation in this build is `planned` except where noted at the bottom; anything
absent from the registry is rejected with `registry.not_registered`.

## Arithmetic (`family: arithmetic`)

| Code | Params | Effect |
| --- | --- | --- |
| `w.arithmetic.linear_interpolate` | `t` (req), `select`, `partial`, `report_per_tensor` | param_transform |
| `w.arithmetic.slerp` | `t` (req), `preserve_norm` | param_transform |
| `w.arithmetic.extrapolate` | `factor` (req) | param_transform |
| `w.arithmetic.delta` | `select`, `partial`, `report_per_tensor` | param_transform |
| `w.arithmetic.scaled_add` | `scale` (req) | param_transform |
| `w.arithmetic.hadamard` | — | param_transform |
| `w.arithmetic.scale` | `factor` (req) | param_transform |
| `w.arithmetic.add_scalar` | `value` (req) | param_transform |
| `w.arithmetic.negate` | — | param_transform |

Assumptions: identical parameter names and shapes (or explicit `partial=True`),
compatible dtypes, finite values. Weight-only: structure digest is preserved.

## Merging (`family: merging`)

| Code | Params | Assumptions |
| --- | --- | --- |
| `w.merge.soup` | `weights`, `density` n/a | same initialization lineage, identical shapes |
| `w.merge.task_arithmetic` | `weights` (first input is the base) | shared base model |
| `w.merge.ties` | `density`, `weights` | shared base; sign conflicts are trimmed by magnitude |
| `w.merge.dare` | `density`, `weights` | shared base; seeded random drop + rescale |
| `w.merge.disjoint` | `weights` | disjoint support across inputs |

Merging is only defensible when parameterizations are compatible. Use NPA alignment
first when in doubt; the registry does not silently align for you.

## Decomposition / representation (`family: decomposition`, `representation`, `structural`)

| Code | Params | Notes |
| --- | --- | --- |
| `w.decomp.svd_truncate` | `rank` or `energy` | real SVD; reports per-tensor energy captured |
| `w.decomp.factorize` | `rank` | produces `.factor_a` / `.factor_b` tensors (rank-r reshape) |
| `w.decomp.orthogonal_transform` | `method` (qr/hadamard), `side`, `scale` | preserves singular values |
| `w.decomp.permute` | `axis`, `permutation` | deterministic permutation |
| `w.decomp.sparsify` | `fraction` (req), `mode` | magnitude or random pruning |

## Structural (`family: structural`)

| Code | Params | Notes |
| --- | --- | --- |
| `w.structural.tie` | `names` (req), `strategy` (mean/first/last) | parameter sharing |
| `w.structural.slice` | `name`, `axis`, `start/stop/step` | changes shape of one tensor |
| `w.structural.extract` | `layers` (req), `keep` | layer/component extraction |
| `w.structural.prune_heads` | `name`, `n_heads`, `heads` | attention-head pruning |
| `w.structural.transplant` | `mapping` (req) | component transplantation between packages |

Tensors whose shapes change are reported; callers must re-validate the target graph.

## Generation (`family: generation`)

| Code | Params | Origin labels |
| --- | --- | --- |
| `w.gen.structured` | `distribution` (orthogonal/normal/uniform/xavier/zeros/ones), `std` | `randomly_initialized` / `analytically_derived` |
| `w.gen.compose_donors` | `conflict` (error/first/last/mean) | `inherited` / `transformed` |
| `w.gen.constraint_matched` | — | `analytically_derived` (moments matched to a template) |
| `w.gen.from_factors` | `suffix_a`, `suffix_b` | `analytically_derived` (rank reconstruction) |
| `w.gen.low_rank_regeneration` | `rank` (req) | `transformed` from a donor |

Generation never claims learned intelligence: random tensors are labelled
`randomly_initialized`, analytical ones `analytically_derived`.

## Reconstruction (`family: reconstruction`)

| Code | Params | Notes |
| --- | --- | --- |
| `w.recon.fill_missing` | `allow_random_fill`, `distribution` | donor-first; fails if nothing can fill and random fill is disallowed |
| `w.recon.low_rank_fill` | `rank` | reconstruct missing tensors from a low-rank donor basis |
| `w.recon.rename` | `mapping` (req) | rebind names without changing values |

## Alignment (`family: alignment`)

| Code | Params | Method |
| --- | --- | --- |
| `w.align.identity` | `threshold`, `allow_partial` | shape/name correspondence only |
| `w.align.procrustes` | `threshold` | closed-form orthogonal Procrustes (SVD) |
| `w.align.linear_map` | `threshold` | ridge least-squares map |
| `w.align.permutation` | `threshold` | Hungarian assignment on cosine cost |
| `w.align.basis` | `threshold` | whitening/colouring basis change |
| `w.align.auto` | `threshold` | evaluate candidates, pick the best defensible one |

Below threshold → `NoValidAlignment` (`npa.no_valid_alignment`). Reports always state
`semantic_equivalence_claimed: false`.

For activation-space comparison use `ParameterAligner(method="activation", graph=..., activation_inputs=...)`
(linear CKA + neuron matching per node).

## Search (`family: search`)

| Code | Algorithm | Dimensional limit |
| --- | --- | --- |
| `w.search.cma_es` | CMA-ES | any low-dimensional space |
| `w.search.evolution_strategy` | (μ, λ) ES | any |
| `w.search.bayesian_optimization` | GP + EI | any |
| `w.search.coordinate_descent` | pattern search | any |
| `w.search.simulated_annealing` | Metropolis | any |
| `w.search.beam_search` | grid beam | ≤ 4 dims (explicit error above) |
| `w.search.nsga2` | NSGA-II | multi-objective: requires `extra_objectives`, returns the Pareto front |

Search operates on low-dimensional transformation parameters, never on billions of
individual weights. Budget exhaustion raises `budget.exhausted`. NSGA-II is kept in a
separate `MULTI_OBJECTIVE_ALGORITHMS` registry, so scalar planning never selects a
multi-objective optimizer by accident and the scalar algorithms keep their
single-objective contract.

## Architecture (NASE / NIR)

| Capability | Status |
| --- | --- |
| Families: `transformer_decoder`, `mlp`, `cnn`, `recurrent` (RNN/LSTM/GRU), `autoencoder`, `moe`, `state_space`, `gnn`, `vae`, `diffusion`, `multimodal`, `hybrid` | implemented |
| NIR operators (39): linear, embedding, pos_embedding, layernorm, rmsnorm, attention, split_qkv, merge_heads, conv2d, pool2d, rnn, lstm, gru, reshape, transpose, concat, slice, flatten, `select`, `moe`, `ssm`, `gcn`, `sample_normal`, `add_noise`, reductions, activations | implemented |
| Graph validation (refs, acyclicity, binding, shapes, dtypes, declared port order) | implemented |
| Composition (sequential/parallel/residual), operator substitution, bypass, truncation | implemented |
| Explicit graph cycles | unsupported by design: recurrent/stateful structure lives inside `rnn`/`lstm`/`gru`/`ssm`, and a cycle is diagnosed rather than silently reinterpreted |
| Sparse-dispatch MoE (per-expert gather/scatter of routed tokens) | implemented; unused experts are never evaluated, with equivalence and NaN-isolation tests |
| Prometheus `/metrics` (counters/gauges/histograms) + JSON logging | implemented; dependency-free registry and scraper endpoint |
| Embedded ONNX runtime, native kernels | planned |

## Runtimes

| Runtime | Status |
| --- | --- |
| NumPy reference (safetensors + npz, graph execution) | implemented |
| PyTorch adapter (weights-only state-dict IO, bf16 widening, `inference_mode` forward) | implemented and verified against `torch 2.15.0+cpu` (CPU only; CUDA availability is reported, never claimed); otherwise typed `runtime.unavailable` |
| Torch-CPU NIR executor (all 39 operators) | implemented; parity-tested against the NumPy reference across all 12 architecture families, with NumPy fallback for unsupported operators |
| GGUF v3 container + GGML type registry (F32/F16; Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q6_K codecs; remaining K-quant/IQ metadata) | implemented: Q4_K/Q6_K are ported from upstream and C-reference validated; other K-quant/IQ tensors are inspectable without decoding and refused for reads by type name/id |
| ONNX interchange for the NIR subset (export/import) | implemented when the `onnx` extra is installed; validated with `onnx.checker`, no embedded executor |
| CUDA-native kernels | planned — UZEN never claims GPU acceleration it does not have |

## Formats

| Format | Status |
| --- | --- |
| `.safetensors` (own codec, strict header validation) | implemented |
| `.npz` | implemented (NumPy runtime) |
| `.pt/.pth/.bin/.ckpt` | PyTorch adapter with `weights_only=True`; pickle objects refused |
| `.pkl` and other pickle formats | **unsupported by design** (`format.unsupported`) |
| `.gguf` (v3 container) | implemented: F32/F16 exact; Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q6_K read (dequantized); Q8_0 write; registry-driven inspection; requantization with error report; remaining K-quant/IQ refused by code |
| `.onnx` | implemented for the exportable NIR subset (requires `onnx`); unmapped operators are refused |

## Interfaces

| Interface | Status |
| --- | --- |
| Python SDK (local + remote HTTP) | implemented |
| CLI (argparse) | implemented |
| REST API (FastAPI, OpenAPI) | implemented (optional extra) |
| MCP stdio server (13 tools) | implemented |
| Worker protocol + daemon (local and remote) | implemented |
| Object-store artifact transport | planned (shared filesystem adapter implemented) |
| Redis/durable queue adapter | planned (SQLite-backed lease queue implemented) |

## Known limitations (explicit)

1. **No distributed scheduler.** The job service claims jobs with SQLite leases; a
   multi-node deployment needs a shared database and an object store.
2. **Remote artifact transfer** between control plane and worker currently assumes a
   shared filesystem or inline tensors.
3. **LLM planning** requires an OpenAI-compatible endpoint; without one it raises rather
   than pretending an LLM was consulted.
4. **Activation alignment** requires a runnable NIR graph and real inputs; it will not
   fabricate activations for unsupported architectures.
5. **Architecture families beyond the five implemented** are reported `planned`; UZEN
   will not synthesize a graph it cannot execute.
