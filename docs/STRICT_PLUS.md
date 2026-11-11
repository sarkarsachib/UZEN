# STRICT_PLUS

STRICT_PLUS is UZEN's foundational execution policy, not a configuration flag.

**Prohibited, everywhere:** conventional training, fine-tuning, backpropagation,
gradient descent, gradient-based parameter updates, optimizer state, data-driven
training loops.

**Permitted:** import, inheritance, extraction, interpolation/extrapolation, task
arithmetic, merging, SVD/low-rank factorization, orthogonal/basis transforms,
permutation alignment, structured pruning, analytical and structured-random
generation, data-free calibration, forward-pass experiments, derivative-free search,
closed-form optimization, architecture synthesis.

## Effects

| Effect | Meaning | Allowed |
| --- | --- | --- |
| `analysis` | read-only inspection/measurement | yes |
| `param_transform` | deterministic math on existing parameters | yes |
| `param_generate` | construct parameters from a stated basis | yes |
| `param_select` | choose among candidate parameter sets | yes |
| `arch_mutation` | change the computational graph | only in architecture scopes |
| `gradient_update` | gradient-based parameter change | **never** |
| `optimizer_state` | optimizer bookkeeping | **never** |
| `data_training` | training on a dataset | **never** |

## Task scopes

`inspection`, `weight_generation`, `weight_modification`, `weight_reconstruction`,
`weight_optimization`, `weight_architecture_research`, `model_creation`,
`model_modification`, `model_reconstruction`, `model_architecture_research`,
`architecture_synthesis`.

Each scope authorizes a fixed set of effects. A weight-modification job cannot mutate a
graph; an architecture job cannot transform parameters. Widening a scope requires a new
job — there is no silent expansion.

## Enforcement points

1. **Policy engine** (`StrictPlusEngine.evaluate/authorize`) — effect matrix, prohibited
   keyword scanning, maturity gating (`planned` is not executable), scope enforcement.
2. **Operation registry** — refuses unregistered codes, non-executable maturity, and
   out-of-scope invocation before any tensor math.
3. **Planner (ANE)** — plans are authorized stage by stage; a prohibited proposal fails
   the whole plan and is audited as `plan_rejected`.
4. **Worker / job service** — the persisted request carries `policy: "STRICT_PLUS"`;
   any other policy string is rejected at submission.
5. **Tensor mutation guard** — refuses writes to tensors with active gradient tracking.
6. **Artifact validation** — policy decisions are stored in the manifest and re-checked
   by VEL; artifacts lacking them are flagged.
7. **LLM bridge** — model proposals pass through 1–3; the model has no authority.

## Audit log

Every decision is appended to a hash-chained log: each entry stores
`prev_hash`, `payload` and `hash = SHA256(prev_hash || canonical(payload))`.
`AuditLog.verify()` detects tampering or truncation. Decisions record the operation
code, effect, scope, maturity, verdict and reason.

## What this means for results

* Random initialization is labelled `randomly_initialized` in every tensor's metadata
  and in generated artifact notes. UZEN never presents random weights as learned
  intelligence.
* A job may legitimately produce a *worse* model. VEL reports measurements; it does not
  promise improvement.
* When a requested transformation is mathematically indefensible (no alignment, shape
  incompatibility, unsupported format), UZEN returns a typed error or
  `NO_VALID_ALIGNMENT` — never a plausible-looking artifact.
