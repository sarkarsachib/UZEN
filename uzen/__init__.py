"""UZEN — Universal Neural Weight Engineering & Architecture Research System.

A headless computational substrate for engineering neural network parameters and
architectures without conventional training. Everything is accessible
programmatically: REST, MCP, CLI and the Python SDK; there is no mandatory frontend.

Quick start (local, in-process)::

    from uzen import Uzen
    client = Uzen(root="./uzen-workspace")
    info = client.import_weights("model.safetensors")
    print(client.inspect("model.safetensors")["family_guess"])
"""

from __future__ import annotations

from uzen import errors
from uzen.errors import UzenError
from uzen.policies.strict_plus import (
    EngineEffect,
    Maturity,
    Origin,
    StrictPlusEngine,
    TaskScope,
    default_engine,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "errors",
    "UzenError",
    "EngineEffect",
    "Maturity",
    "Origin",
    "TaskScope",
    "StrictPlusEngine",
    "default_engine",
    "Uzen",
]


def __getattr__(name: str):
    # Lazy import keeps `import uzen` cheap and avoids pulling the full engine in
    # for callers that only need policy primitives.
    if name == "Uzen":
        from uzen.interfaces.python_sdk.client import Uzen

        return Uzen
    if name == "JobService":
        from uzen.jobs.service import JobService

        return JobService
    if name == "WeightPackage":
        from uzen.core.nws.tensors import WeightPackage

        return WeightPackage
    raise AttributeError(f"module 'uzen' has no attribute '{name}'")
