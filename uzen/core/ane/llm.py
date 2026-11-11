"""LLM-assisted planning for ANE.

The autonomous engineer can optionally consult an OpenAI-compatible chat endpoint to
propose an engineering plan. Two invariants keep this safe and honest:

1. **The LLM cannot authorize.** Every proposed stage is resolved against the
   operation registry, parameter-validated, scope-checked and policy-authorized
   exactly like a deterministic plan. Prohibited proposals (e.g. "fine-tune with
   AdamW") are rejected with a typed error.
2. **No silent fallback.** If no endpoint is configured, the planner raises
   :class:`uzen.errors.UzenError` rather than pretending an LLM was consulted.

Configuration (environment):

* ``UZEN_LLM_BASE_URL`` — e.g. ``https://api.openai.com/v1``
* ``UZEN_LLM_API_KEY`` — bearer token (never logged)
* ``UZEN_LLM_MODEL`` — model name
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from uzen.core.ane.planner import EngineeringPlan, JobRequest, PlanStage
from uzen.core.nwpc.registry import OperationRegistry, default_registry
from uzen.errors import InvalidOperationParameters, UzenError
from uzen.policies.strict_plus.engine import StrictPlusEngine, default_engine

SYSTEM_PROMPT = """You are the planning component of UZEN, a headless neural weight
engineering system operating under the STRICT_PLUS policy.

STRICT_PLUS forbids: conventional training, fine-tuning, backpropagation, gradient
descent, gradient-based parameter updates, optimizer state.

You may only propose stages from the operation catalogue provided. Respond with JSON:
{"rationale": ["..."], "stages": [{"name": "...", "op": "<operation code>",
 "args": ["<input or earlier stage name>"], "params": {...}}],
 "search_space": [{"stage": "...", "param": "...", "low": 0.0, "high": 1.0}]}

Rules:
- Use only operation codes from the catalogue.
- Only reference declared input names or outputs of earlier stages.
- Never propose training, fine-tuning, gradients, or optimizers.
- If no catalogue operation can achieve the objective, return {"stages": []} and say
  so in the rationale.
"""


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "LLMConfig | None":
        base = os.environ.get("UZEN_LLM_BASE_URL")
        key = os.environ.get("UZEN_LLM_API_KEY")
        model = os.environ.get("UZEN_LLM_MODEL")
        if not (base and key and model):
            return None
        return cls(base_url=base.rstrip("/"), api_key=key, model=model)


class LLMPlanner:
    """Optional AI planner. Proposals are validated by the registry and policy engine."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        registry: OperationRegistry | None = None,
        engine: StrictPlusEngine | None = None,
    ) -> None:
        self.config = config or LLMConfig.from_env()
        self.registry = registry or default_registry()
        self.engine = engine or default_engine()

    @property
    def configured(self) -> bool:
        return self.config is not None

    # ------------------------------------------------------------------ #

    def propose(self, request: JobRequest, input_names: Sequence[str]) -> EngineeringPlan:
        if self.config is None:
            raise UzenError(
                "no LLM endpoint is configured; set UZEN_LLM_BASE_URL, UZEN_LLM_API_KEY and "
                "UZEN_LLM_MODEL to enable AI-assisted planning. The deterministic planner "
                "remains available and is the default."
            )
        catalogue = self._catalogue_prompt()
        user_prompt = json.dumps(
            {
                "objective": request.objective,
                "task_scope": request.task_scope.value,
                "inputs": list(input_names),
                "constraints": request.constraints,
                "evaluation_plan": request.evaluation_plan,
                "catalogue": catalogue,
            },
            indent=2,
        )
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        response = self._post("/chat/completions", payload)
        try:
            content = response["choices"][0]["message"]["content"]
            proposal = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise UzenError(
                "the LLM response was not valid plan JSON", response_excerpt=str(response)[:400]
            ) from exc
        return self._validate(request, input_names, proposal)

    # ------------------------------------------------------------------ #

    def _catalogue_prompt(self) -> list[dict[str, Any]]:
        catalogue = []
        for spec in self.registry.all():
            if not spec.maturity.executable:
                continue
            if spec.effect.value in ("gradient_update", "optimizer_state", "data_training"):
                continue
            catalogue.append(
                {
                    "code": spec.code,
                    "title": spec.title,
                    "effect": spec.effect.value,
                    "inputs": f"{spec.min_inputs}..{spec.max_inputs}",
                    "params": {
                        name: {"type": p.type, "required": p.required, "default": p.default}
                        for name, p in spec.params.items()
                    },
                }
            )
        return catalogue

    def _validate(
        self,
        request: JobRequest,
        input_names: Sequence[str],
        proposal: Mapping[str, Any],
    ) -> EngineeringPlan:
        stages_in = proposal.get("stages") or []
        if not stages_in:
            raise UzenError(
                "the LLM did not propose any executable stage for this objective",
                rationale=proposal.get("rationale", []),
            )
        search_entries = {entry.get("stage"): entry for entry in proposal.get("search_space", [])}
        known = set(input_names)
        stages: list[PlanStage] = []
        rationale = [str(r) for r in proposal.get("rationale", [])]
        for entry in stages_in:
            code = str(entry.get("op", ""))
            spec = self.registry.get(code)  # raises OperationNotRegistered
            if not spec.maturity.executable:
                raise InvalidOperationParameters(
                    f"LLM proposed non-executable operation '{code}' ({spec.maturity.value})"
                )
            if spec.effect.value in ("gradient_update", "optimizer_state", "data_training"):
                raise UzenError(
                    f"LLM proposed a prohibited effect ({spec.effect.value}); STRICT_PLUS "
                    "rejects this plan",
                    operation=code,
                )
            decision = self.engine.evaluate(spec.effect, code, scope=request.task_scope)
            if not decision.allowed:
                raise UzenError(
                    f"LLM-proposed stage '{code}' is not authorized: {decision.reason}"
                )
            params = spec.validate_params(dict(entry.get("params") or {}))
            args = [str(a) for a in entry.get("args", [])]
            for arg in args:
                if arg not in known:
                    raise InvalidOperationParameters(
                        f"LLM proposed a stage referencing unknown value '{arg}'"
                    )
            stage_name = str(entry.get("name") or f"stage{len(stages)}")
            search_space: list[tuple[str, float, float]] = []
            entry_search = search_entries.get(stage_name)
            if entry_search:
                search_space.append(
                    (
                        str(entry_search["param"]),
                        float(entry_search["low"]),
                        float(entry_search["high"]),
                    )
                )
            stages.append(
                PlanStage(
                    name=stage_name,
                    op_code=code,
                    input_names=args or list(input_names[:1]),
                    fixed_params=params,
                    search_space=search_space,
                    description=str(entry.get("description", "LLM-proposed stage")),
                    effect=spec.effect,
                )
            )
            known.add(stage_name)
        return EngineeringPlan(
            request=request,
            stages=stages,
            rationale=["LLM-proposed plan, validated by the registry and STRICT_PLUS"] + rationale,
            warnings=[
                "plan was generated by an LLM and accepted only because every stage is a "
                "registered, policy-authorized operation"
            ],
            search_algorithm=str(request.budget.get("search_algorithm", "coordinate_descent")),
            evaluation=dict(request.evaluation_plan),
            candidate_budget=int(request.budget.get("candidates", 8)),
            evaluations_per_stage=int(request.budget.get("evaluations", 16)),
        )

    # ------------------------------------------------------------------ #

    def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        assert self.config is not None
        url = f"{self.config.base_url}{path}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise UzenError(
                f"LLM endpoint returned HTTP {exc.code}", body=body, url=url
            ) from exc
        except urllib.error.URLError as exc:
            raise UzenError(f"cannot reach LLM endpoint: {exc.reason}", url=url) from exc
