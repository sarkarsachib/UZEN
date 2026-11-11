"""Lineage graph handling.

Artifacts form a DAG: each artifact records its parents and the operation chain
that transformed them. This module verifies DAG-ness, resolves ancestry and
produces human/agent-readable lineage reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from uzen.errors import ArtifactIntegrityError


@dataclass
class LineageNode:
    artifact_id: str
    parents: list[str] = field(default_factory=list)
    operation: str = ""
    operations: list[dict] = field(default_factory=list)
    created_at: str = ""
    family: str = "generic"


class LineageGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, LineageNode] = {}

    def add(self, node: LineageNode) -> None:
        existing = self.nodes.get(node.artifact_id)
        if existing and existing.parents != node.parents:
            raise ArtifactIntegrityError(
                f"artifact {node.artifact_id} already registered with different parents"
            )
        self.nodes[node.artifact_id] = node

    def children(self, artifact_id: str) -> list[str]:
        return [nid for nid, n in self.nodes.items() if artifact_id in n.parents]

    def ancestors(self, artifact_id: str, *, max_depth: int = 64) -> list[str]:
        seen: set[str] = set()
        frontier = [artifact_id]
        depth = 0
        while frontier and depth < max_depth:
            depth += 1
            nxt: list[str] = []
            for nid in frontier:
                node = self.nodes.get(nid)
                if node is None:
                    continue
                for p in node.parents:
                    if p not in seen:
                        seen.add(p)
                        nxt.append(p)
            frontier = nxt
        return sorted(seen)

    def path_to(self, artifact_id: str) -> list[str]:
        """Return a root-first operation path (longest-parent chain heuristic)."""
        chain: list[str] = []
        current = artifact_id
        guard = 0
        while current in self.nodes and guard < 512:
            chain.append(current)
            parents = self.nodes[current].parents
            if not parents:
                break
            current = parents[0]
            guard += 1
        return list(reversed(chain))

    def validate(self) -> tuple[bool, list[str]]:
        """Verify the graph is acyclic and every referenced parent is known."""
        problems: list[str] = []
        for nid, node in self.nodes.items():
            for p in node.parents:
                if p not in self.nodes:
                    problems.append(f"{nid}: missing parent {p}")
                if p == nid:
                    problems.append(f"{nid}: self-referential lineage")
        # Cycle detection via DFS colouring.
        WHITE, GREY, BLACK = 0, 1, 2
        colour: dict[str, int] = {nid: WHITE for nid in self.nodes}

        def visit(nid: str, stack: list[str]) -> None:
            colour[nid] = GREY
            for p in self.nodes[nid].parents:
                if p not in self.nodes:
                    continue
                if colour[p] == GREY:
                    cycle = " -> ".join(stack + [nid, p])
                    problems.append(f"cycle detected: {cycle}")
                    return
                if colour[p] == WHITE:
                    visit(p, stack + [nid])
            colour[nid] = BLACK

        for nid in list(self.nodes):
            if colour[nid] == WHITE:
                visit(nid, [])
        return (not problems, problems)

    def to_dict(self) -> dict:
        return {
            "nodes": {
                nid: {
                    "parents": n.parents,
                    "operation": n.operation,
                    "operations": n.operations,
                    "created_at": n.created_at,
                    "family": n.family,
                }
                for nid, n in self.nodes.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "LineageGraph":
        g = cls()
        for nid, node in (data.get("nodes") or {}).items():
            g.add(
                LineageNode(
                    artifact_id=nid,
                    parents=list(node.get("parents", [])),
                    operation=str(node.get("operation", "")),
                    operations=list(node.get("operations", [])),
                    created_at=str(node.get("created_at", "")),
                    family=str(node.get("family", "generic")),
                )
            )
        return g

    def describe(self, artifact_id: str) -> dict:
        chain = self.path_to(artifact_id)
        # ``chain`` is root-first and includes the full node record, so a consumer does
        # not have to re-resolve ids against the graph to describe the derivation.
        described = [
            {
                "artifact_id": nid,
                "operation": self.nodes[nid].operation,
                "family": self.nodes[nid].family,
                "parents": list(self.nodes[nid].parents),
                "created_at": self.nodes[nid].created_at,
            }
            for nid in chain
        ]
        return {
            "artifact_id": artifact_id,
            "chain": described,
            "ancestors": self.ancestors(artifact_id),
            "children": self.children(artifact_id),
            "operations": [
                {"artifact": nid, "operation": self.nodes[nid].operation} for nid in chain
            ],
        }
