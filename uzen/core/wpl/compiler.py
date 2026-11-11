"""WPL compiler: program AST -> typed execution plan -> executed artifacts.

The compiler performs four checks before anything runs:

1. **Operation resolution** — every called operation exists in the registry, either
   by full code (``w.arithmetic.delta``) or by unambiguous shorthand (``delta``).
2. **Parameter typing** — every argument is validated against the operation's
   declared parameter spec; unknown parameters are rejected.
3. **Reference/dataflow checking** — a step may only reference declared inputs or
   outputs of earlier steps.
4. **Policy authorization** — each stage's declared effect is authorized for the
   program's scope *before* execution (STRICT_PLUS dry run).

Execution is delegated to :class:`uzen.core.wcomp.compiler.CompiledPipeline`, which
calls the operation registry. There is no interpreter that evaluates arbitrary code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from uzen.core.nwpc.registry import OperationRegistry, default_registry
from uzen.core.wcomp.compiler import (
    CompiledPipeline,
    CompilationReport,
    PipelineResult,
    Stage,
    compile_stages,
)
from uzen.core.wpl.parser import Call, Program, parse_program
from uzen.errors import ProgramInvalid
from uzen.policies.strict_plus.effects import TaskScope
from uzen.policies.strict_plus.engine import StrictPlusEngine, default_engine

TYPE_MAP = {"float": float, "int": int, "bool": bool, "str": str}


@dataclass
class CompiledProgram:
    program: Program
    pipeline: CompiledPipeline
    report: CompilationReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program.name,
            "inputs": list(self.program.inputs),
            "params": {k: {"type": v[0], "default": v[1]} for k, v in self.program.params.items()},
            "outputs": dict(self.program.outputs),
            "compilation": self.report.to_dict(),
            "effects": self.pipeline.effects(),
        }


def resolve_operation_code(code: str, registry: OperationRegistry) -> str:
    """Resolve a full code or a unique shorthand into a registered operation code."""
    if code.startswith("w."):
        registry.get(code)  # raises if unknown
        return code
    matches = [spec.code for spec in registry.all() if spec.code.split(".")[-1] == code]
    if not matches:
        matches = [spec.code for spec in registry.all() if code in spec.code]
    if not matches:
        raise ProgramInvalid(
            f"unknown operation '{code}' in WPL program; see GET /v1/operations for the catalogue"
        )
    if len(matches) > 1:
        raise ProgramInvalid(
            f"operation shorthand '{code}' is ambiguous: {matches}; use the full code"
        )
    return matches[0]


def _resolve_value(value: Any, params: Mapping[str, tuple[str, Any]]) -> Any:
    if isinstance(value, str) and value in params and not value.startswith("w."):
        return params[value][1]
    return value


def compile_program(
    source: str,
    *,
    scope: TaskScope = TaskScope.WEIGHT_MODIFICATION,
    registry: OperationRegistry | None = None,
    engine: StrictPlusEngine | None = None,
    param_overrides: Mapping[str, Any] | None = None,
) -> CompiledProgram:
    """Parse and compile WPL source into an executable pipeline."""
    program = parse_program(source)
    registry = registry or default_registry()
    engine = engine or default_engine()

    params: dict[str, tuple[str, Any]] = {}
    overrides = dict(param_overrides or {})
    unknown_overrides = set(overrides) - set(program.params)
    if unknown_overrides:
        raise ProgramInvalid(
            f"override for undeclared program parameters: {sorted(unknown_overrides)}"
        )
    for name, (ptype, default) in program.params.items():
        value = overrides.get(name, default)
        if value is None:
            raise ProgramInvalid(
                f"program parameter '{name}' has no default and no override was supplied"
            )
        expected = TYPE_MAP[ptype]
        if expected is bool:
            ok = isinstance(value, bool)
        elif expected is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif expected is float:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ProgramInvalid(
                    f"program parameter '{name}' must be float, got {value!r}"
                ) from exc
        else:
            ok = isinstance(value, str)
        if not ok:
            raise ProgramInvalid(
                f"program parameter '{name}' must be {ptype}, got {type(value).__name__}"
            )
        params[name] = (ptype, value)

    stages: list[Stage] = []
    for name, call in program.steps:
        code = resolve_operation_code(call.code, registry)
        kwargs = {k: _resolve_value(v, params) for k, v in call.kwargs.items()}
        stages.append(Stage(name=name, code=code, args=list(call.args), params=kwargs))

    pipeline, report = compile_stages(
        stages,
        scope=scope,
        registry=registry,
        engine=engine,
        available_inputs=program.inputs,
        outputs=list(program.outputs.values()),
        metadata={
            "program": program.name,
            "outputs": sorted(set(program.outputs.values())),
        },
    )
    report.require()
    return CompiledProgram(program=program, pipeline=pipeline, report=report)


def run_program(
    source: str,
    inputs: Mapping[str, Any],
    *,
    scope: TaskScope = TaskScope.WEIGHT_MODIFICATION,
    seed: int | str | None = None,
    registry: OperationRegistry | None = None,
    engine: StrictPlusEngine | None = None,
    param_overrides: Mapping[str, Any] | None = None,
    job_id: str | None = None,
) -> tuple[PipelineResult, CompiledProgram]:
    compiled = compile_program(
        source,
        scope=scope,
        registry=registry,
        engine=engine,
        param_overrides=param_overrides,
    )
    result = compiled.pipeline.execute(inputs, seed=seed, scope=scope, job_id=job_id)
    return result, compiled


def describe_program(source: str, *, scope: TaskScope | None = None, registry: OperationRegistry | None = None) -> dict[str, Any]:
    """Static analysis of a program without executing it."""
    program = parse_program(source)
    registry = registry or default_registry()
    stages = []
    effects: list[str] = []
    for name, call in program.steps:
        code = resolve_operation_code(call.code, registry)
        spec = registry.get(code)
        effects.append(spec.effect.value)
        stages.append(
            {
                "name": name,
                "code": code,
                "title": spec.title,
                "effect": spec.effect.value,
                "maturity": spec.maturity.value,
                "args": list(call.args),
                "params": dict(call.kwargs),
                "assumptions": list(spec.assumptions),
            }
        )
    return {
        "program": program.name,
        "inputs": program.inputs,
        "params": {k: {"type": v[0], "default": v[1]} for k, v in program.params.items()},
        "steps": stages,
        "outputs": program.outputs,
        "effects": sorted(set(effects)),
        "scope_requested": scope.value if scope else None,
        "executable": True,
    }
