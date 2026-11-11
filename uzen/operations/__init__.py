"""Weight-engineering operation catalogue.

Importing this package registers every operation in the process-wide registry.
Modules are imported in dependency order; each registers itself with
:func:`uzen.core.nwpc.registry.default_registry`.
"""

from __future__ import annotations

from uzen.operations import (  # noqa: F401
    alignment,
    arithmetic,
    decomposition,
    generation,
    merging,
    reconstruction,
    search,
    structural,
)

__all__ = [
    "alignment",
    "arithmetic",
    "decomposition",
    "generation",
    "merging",
    "reconstruction",
    "search",
    "structural",
]
