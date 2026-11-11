"""WPL language, compiler and pipeline tests."""

from __future__ import annotations

import numpy as np
import pytest

from uzen.core.nws.tensors import WeightPackage
from uzen.core.wcomp.compiler import compile_stages
from uzen.core.wpl.compiler import compile_program, describe_program, run_program
from uzen.core.wpl.parser import parse_program, tokenize
from uzen.errors import ProgramInvalid
from uzen.policies.strict_plus.effects import TaskScope


PROGRAM = """
program "blend_and_scale" {
    input base: package
    input other: package
    param t: float = 0.5
    param factor: float = 1.0
    step blend = linear_interpolate(base, other, t=t)
    step out = scale(blend, factor=factor)
    output result = out
}
"""


def _packages():
    a = WeightPackage.from_arrays({"w": np.zeros((2, 2), dtype=np.float32)}, name="a")
    b = WeightPackage.from_arrays({"w": np.full((2, 2), 4.0, dtype=np.float32)}, name="b")
    return a, b


def test_tokenizer_handles_structure_and_comments():
    tokens = tokenize("# comment\nprogram \"p\" {\n  input x: package\n}\n")
    kinds = [t.kind for t in tokens]
    assert "IDENT" in kinds and "PUNCT" in kinds
    assert tokens[-1].kind == "EOF"
    assert all(t.line >= 1 for t in tokens)


def test_parser_builds_program_ast():
    program = parse_program(PROGRAM)
    assert program.name == "blend_and_scale"
    assert program.inputs == ["base", "other"]
    assert [name for name, _ in program.steps] == ["blend", "out"]
    assert program.outputs["result"] == "out"


def test_parser_rejects_malformed_source():
    with pytest.raises(ProgramInvalid):
        parse_program('program "broken" { step x = scale( }')
    with pytest.raises(ProgramInvalid):
        parse_program("")
    with pytest.raises(ProgramInvalid):
        parse_program('program "no_steps" { input a: package }')


def test_parser_requires_package_inputs():
    with pytest.raises(ProgramInvalid) as exc:
        parse_program('program "bad" { input a: weights step s = scale(a) }')
    assert "package" in str(exc.value)


def test_compile_resolves_operations_and_types_parameters():
    compiled = compile_program(PROGRAM)
    assert [s.code for s in compiled.pipeline.stages] == [
        "w.arithmetic.linear_interpolate",
        "w.arithmetic.scale",
    ]
    assert compiled.report.ok
    assert compiled.pipeline.stages[0].params["t"] == 0.5


def test_compile_rejects_bad_parameter_type_and_unknown_override():
    with pytest.raises(ProgramInvalid):
        compile_program(PROGRAM, param_overrides={"t": "not-a-number"})
    with pytest.raises(ProgramInvalid):
        compile_program(PROGRAM, param_overrides={"unknown": 1})


def test_compile_rejects_unknown_operation():
    bad = """
    program "bad" {
      input a: package
      step x = no_such_operation(a)
      output o = x
    }
    """
    with pytest.raises(Exception) as exc:
        compile_program(bad)
    assert "no_such_operation" in str(exc.value)


def test_program_executes_real_math():
    a, b = _packages()
    result, compiled = run_program(PROGRAM, {"base": a, "other": b}, seed=0)
    package = result.package()
    np.testing.assert_allclose(package.tensors["w"], np.full((2, 2), 2.0, dtype=np.float32))
    assert package.structure_digest() == a.structure_digest()
    assert compiled.report.ok


def test_program_overrides_change_result():
    a, b = _packages()
    result, _ = run_program(
        PROGRAM, {"base": a, "other": b}, param_overrides={"t": 1.0, "factor": 0.25}, seed=1
    )
    np.testing.assert_allclose(result.package().tensors["w"], np.full((2, 2), 1.0, dtype=np.float32))


def test_program_reports_missing_input():
    a, _ = _packages()
    with pytest.raises(Exception):
        run_program(PROGRAM, {"base": a}, seed=0)


def test_policy_applies_to_programs():
    src = """
    program "arch_in_weight_scope" {
      input a: package
      step out = synthesize(a)
      output o = out
    }
    """
    with pytest.raises(Exception):
        compile_program(src, scope=TaskScope.WEIGHT_MODIFICATION)


def test_describe_program_is_static():
    described = describe_program(PROGRAM)
    assert described["program"] == "blend_and_scale"
    assert len(described["steps"]) == 2
    assert all("effect" in s and "maturity" in s for s in described["steps"])


def test_compile_stages_flags_undeclared_inputs():
    from uzen.core.wcomp.compiler import Stage

    _, report = compile_stages(
        [Stage(name="x", code="w.arithmetic.scale", args=["missing"], params={"factor": 2.0})],
        scope=TaskScope.WEIGHT_MODIFICATION,
        available_inputs=["base"],
        outputs=["x"],
    )
    assert report.ok is False
    assert any("missing" in p for p in report.problems)
