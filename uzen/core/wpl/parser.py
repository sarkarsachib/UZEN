"""WPL: the Weight Programming Language.

WPL is a small, real, declaratively-checked language for expressing weight
engineering programs. It is compiled to an execution plan (WComp) and executed by
the operation registry (NWPC) under STRICT_PLUS.

Grammar (EBNF-ish)::

    program     := "program" STRING? "{" statement* "}"
    statement   := input | param | let | output
    input       := "input" IDENT ":" "package"
    param       := "param" IDENT ":" TYPE ("=" literal)?
    let         := "step" IDENT "=" call
    output      := "output" (IDENT "=")? reference
    call        := opcode "(" arglist? ")"
    opcode      := IDENT ("." IDENT)*
    arglist     := arg ("," arg)*
    arg         := IDENT "=" literal | reference
    literal     := NUMBER | STRING | "true" | "false" | list
    reference   := IDENT

Example::

    program "aligned_blend" {
      input base : package
      input donor : package
      param factor : float = 0.35
      param method : str = "procrustes"
      step aligned  = w.align.procrustes(donor, base, threshold=0.4)
      step delta    = w.arithmetic.delta(aligned, base)
      step scaled   = w.arithmetic.scale(delta, factor=factor)
      step combined = w.arithmetic.scaled_add(base, scaled, scale=1.0)
      output weights = combined
    }

The lexer, parser and compiler are implemented here; there is no hidden Python
``eval`` — parameters are literals or declared program parameters only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from uzen.errors import ProgramInvalid

KEYWORDS = {"program", "input", "param", "step", "let", "output", "package", "float", "int", "bool", "str"}


# --------------------------------------------------------------------------- #
# lexer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Token:
    kind: str  # IDENT | NUMBER | STRING | PUNCT | EOF
    value: str
    line: int
    column: int

    def __str__(self) -> str:
        return f"{self.kind}({self.value!r})@{self.line}:{self.column}"


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i, line, col = 0, 1, 1
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in " \t\r":
            i += 1
            col += 1
            continue
        if ch == "\n":
            i += 1
            line += 1
            col = 1
            continue
        if ch == "#":
            while i < n and source[i] != "\n":
                i += 1
                col += 1
            continue
        if ch == '"':
            start_line, start_col = line, col
            i += 1
            col += 1
            buf: list[str] = []
            while i < n and source[i] != '"':
                if source[i] == "\\" and i + 1 < n:
                    nxt = source[i + 1]
                    buf.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(nxt, nxt))
                    i += 2
                    col += 2
                    continue
                if source[i] == "\n":
                    raise ProgramInvalid(f"unterminated string at line {start_line}")
                buf.append(source[i])
                i += 1
                col += 1
            if i >= n:
                raise ProgramInvalid(f"unterminated string starting at line {start_line}")
            i += 1
            col += 1
            tokens.append(Token("STRING", "".join(buf), start_line, start_col))
            continue
        if ch.isdigit() or (ch in "+-" and i + 1 < n and (source[i + 1].isdigit() or source[i + 1] == ".")):
            start = i
            start_col = col
            if source[i] in "+-":
                i += 1
                col += 1
            while i < n and (source[i].isdigit() or source[i] in ".eE"):
                if source[i] in "eE" and i + 1 < n and source[i + 1] in "+-":
                    i += 1
                    col += 1
                i += 1
                col += 1
            tokens.append(Token("NUMBER", source[start:i], line, start_col))
            continue
        if ch.isalpha() or ch == "_":
            start = i
            start_col = col
            while i < n and (source[i].isalnum() or source[i] in "_."):
                i += 1
                col += 1
            tokens.append(Token("IDENT", source[start:i], line, start_col))
            continue
        if ch in "{}()=,:[]":
            tokens.append(Token("PUNCT", ch, line, col))
            i += 1
            col += 1
            continue
        raise ProgramInvalid(f"unexpected character {ch!r} at line {line}, column {col}")
    tokens.append(Token("EOF", "", line, col))
    return tokens


# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #


@dataclass
class Call:
    code: str
    args: list[str] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Program:
    name: str
    inputs: list[str]
    params: dict[str, tuple[str, Any]]
    steps: list[tuple[str, Call]]
    outputs: dict[str, str]


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


class Parser:
    def __init__(self, tokens: Sequence[Token]) -> None:
        self.tokens = list(tokens)
        self.pos = 0

    # -- helpers ------------------------------------------------------- #

    def peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else self.tokens[-1]

    def next(self) -> Token:
        tok = self.peek()
        self.pos += 1
        return tok

    def expect(self, kind: str, value: str | None = None) -> Token:
        tok = self.peek()
        if tok.kind != kind or (value is not None and tok.value != value):
            expected = f"{kind}({value})" if value else kind
            raise ProgramInvalid(
                f"expected {expected} but found {tok} at line {tok.line}, column {tok.column}"
            )
        return self.next()

    def at_punct(self, value: str) -> bool:
        return self.peek().kind == "PUNCT" and self.peek().value == value

    def at_keyword(self, value: str) -> bool:
        return self.peek().kind == "IDENT" and self.peek().value == value

    # -- grammar ------------------------------------------------------- #

    def parse(self) -> Program:
        self.expect("IDENT", "program")
        name = "program"
        if self.peek().kind == "STRING":
            name = self.next().value
        self.expect("PUNCT", "{")
        inputs: list[str] = []
        params: dict[str, tuple[str, Any]] = {}
        steps: list[tuple[str, Call]] = []
        outputs: dict[str, str] = {}
        while not self.at_punct("}"):
            if self.peek().kind == "EOF":
                raise ProgramInvalid(f"unterminated program '{name}': missing '}}'")
            if self.at_keyword("input"):
                inputs.append(self._parse_input())
            elif self.at_keyword("param"):
                pname, ptype, pdefault = self._parse_param()
                params[pname] = (ptype, pdefault)
            elif self.at_keyword("step") or self.at_keyword("let"):
                steps.append(self._parse_step())
            elif self.at_keyword("output"):
                oname, ref = self._parse_output()
                outputs[oname] = ref
            else:
                tok = self.peek()
                raise ProgramInvalid(
                    f"unexpected token {tok} in program body at line {tok.line}"
                )
        self.expect("PUNCT", "}")
        self.expect("EOF")
        if not steps:
            raise ProgramInvalid("program contains no steps")
        if not outputs:
            outputs = {steps[-1][0]: steps[-1][0]}
        return Program(name=name, inputs=inputs, params=params, steps=steps, outputs=outputs)

    def _parse_input(self) -> str:
        self.expect("IDENT", "input")
        name = self.expect("IDENT").value
        self.expect("PUNCT", ":")
        type_tok = self.expect("IDENT")
        if type_tok.value != "package":
            raise ProgramInvalid(
                f"input '{name}' must be declared ': package' (WPL transports parameter "
                f"packages, not raw tensors), found '{type_tok.value}'"
            )
        return name

    def _parse_param(self) -> tuple[str, str, Any]:
        self.expect("IDENT", "param")
        name = self.expect("IDENT").value
        self.expect("PUNCT", ":")
        ptype = self.expect("IDENT").value
        if ptype not in ("float", "int", "bool", "str"):
            raise ProgramInvalid(f"unsupported parameter type '{ptype}'")
        default = None
        if self.at_punct("="):
            self.next()
            default = self._parse_literal()
        return name, ptype, default

    def _parse_step(self) -> tuple[str, Call]:
        self.next()  # 'step' or 'let'
        name = self.expect("IDENT").value
        self.expect("PUNCT", "=")
        call = self._parse_call()
        return name, call

    def _parse_output(self) -> tuple[str, str]:
        self.expect("IDENT", "output")
        if self.peek().kind == "IDENT" and self.peek(1).kind == "PUNCT" and self.peek(1).value == "=":
            name = self.next().value
            self.expect("PUNCT", "=")
            ref = self.expect("IDENT").value
            return name, ref
        ref = self.expect("IDENT").value
        return ref, ref

    def _parse_call(self) -> Call:
        code_parts = [self.expect("IDENT").value]
        while self.at_punct("."):
            # dotted opcodes are lexed as a single IDENT ('.' allowed inside idents),
            # so this branch only fires if the lexer split them.
            self.next()
            code_parts.append(self.expect("IDENT").value)
        code = ".".join(code_parts)
        self.expect("PUNCT", "(")
        args: list[str] = []
        kwargs: dict[str, Any] = {}
        if not self.at_punct(")"):
            while True:
                if (
                    self.peek().kind == "IDENT"
                    and self.peek(1).kind == "PUNCT"
                    and self.peek(1).value == "="
                ):
                    key = self.next().value
                    self.expect("PUNCT", "=")
                    kwargs[key] = self._parse_literal()
                else:
                    args.append(self.expect("IDENT").value)
                if self.at_punct(","):
                    self.next()
                    continue
                break
        self.expect("PUNCT", ")")
        return Call(code=code, args=args, kwargs=kwargs)

    def _parse_literal(self) -> Any:
        tok = self.peek()
        if tok.kind == "NUMBER":
            self.next()
            text = tok.value
            try:
                return int(text) if ("." not in text and "e" not in text.lower()) else float(text)
            except ValueError as exc:
                raise ProgramInvalid(f"invalid number literal {text!r}") from exc
        if tok.kind == "STRING":
            self.next()
            return tok.value
        if tok.kind == "IDENT":
            self.next()
            if tok.value in ("true", "false"):
                return tok.value == "true"
            # Identifier literals are symbolic references to declared program params;
            # the compiler resolves them and rejects anything undeclared.
            return tok.value
        if self.at_punct("["):
            self.next()
            items: list[Any] = []
            if not self.at_punct("]"):
                while True:
                    items.append(self._parse_literal())
                    if self.at_punct(","):
                        self.next()
                        continue
                    break
            self.expect("PUNCT", "]")
            return items
        raise ProgramInvalid(f"expected a literal, found {tok}")


def parse_program(source: str) -> Program:
    if not isinstance(source, str) or not source.strip():
        raise ProgramInvalid("empty WPL source")
    return Parser(tokenize(source)).parse()
