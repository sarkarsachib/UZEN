"""Python SDK for UZEN.

Two transports, one surface:

* ``Uzen.local(root)`` — runs everything in-process against a job service and artifact
  store (no HTTP server required).
* ``Uzen.remote(url, token)`` — talks to a running UZEN REST control plane.

The SDK never fabricates results: local calls execute the real engines, and remote
calls surface the server's typed error payloads as exceptions.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from uzen.core.nws.tensors import WeightPackage
from uzen.errors import UzenError
from uzen.jobs.service import JobService


class UzenClientError(UzenError):
    code = "sdk.error"


@dataclass
class _RemoteConfig:
    base_url: str
    token: str | None = None
    timeout: float = 120.0


class Uzen:
    """Unified SDK entry point."""

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        if (root is None) == (base_url is None):
            raise UzenClientError(
                "provide exactly one of root= (local engine) or base_url= (remote control plane)"
            )
        self._service = JobService(root) if root is not None else None
        self._remote = _RemoteConfig(base_url.rstrip("/"), token, timeout) if base_url else None

    # ------------------------------------------------------------------ #
    # transport
    # ------------------------------------------------------------------ #

    @property
    def local(self) -> bool:
        return self._service is not None

    @property
    def service(self) -> JobService:
        if self._service is None:
            raise UzenClientError("this client is remote; use job endpoints instead")
        return self._service

    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        assert self._remote is not None
        url = f"{self._remote.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self._remote.token:
            req.add_header("Authorization", f"Bearer {self._remote.token}")
        try:
            with urllib.request.urlopen(req, timeout=self._remote.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(payload)
                detail = parsed.get("detail") or parsed
            except json.JSONDecodeError:
                detail = {"message": payload[:500]}
            raise UzenClientError(
                f"control plane returned HTTP {exc.code}", detail=detail, path=path
            ) from exc
        except urllib.error.URLError as exc:
            raise UzenClientError(
                f"cannot reach control plane at {self._remote.base_url}: {exc.reason}", path=path
            ) from exc

    # ------------------------------------------------------------------ #
    # catalogue
    # ------------------------------------------------------------------ #

    def operations(self) -> list[dict[str, Any]]:
        if self.local:
            from uzen.core.nwpc.registry import default_registry

            return default_registry().catalogue()
        return self._request("GET", "/v1/operations")["operations"]

    def policy(self) -> dict[str, Any]:
        if self.local:
            return self.service.engine.describe()
        return self._request("GET", "/v1/policies/strict-plus")

    def families(self) -> list[dict[str, Any]]:
        if self.local:
            from uzen.core.nws.families import family_catalogue

            return family_catalogue()
        return self._request("GET", "/v1/architectures/families")["families"]

    # ------------------------------------------------------------------ #
    # artifacts
    # ------------------------------------------------------------------ #

    def import_weights(self, path: str | Path, *, name: str | None = None, family: str | None = None) -> dict[str, Any]:
        if self.local:
            record = self.service.import_artifact(path, name=name, family=family)
            return {
                "artifact_id": record.artifact_id,
                "name": record.manifest.name,
                "family": record.manifest.family,
                "num_tensors": record.num_tensors,
                "checksum": record.manifest.checksum,
            }
        return self._request(
            "POST", "/v1/models/import", {"path": str(path), "name": name, "family": family}
        )

    def inspect(self, source: str | Path) -> dict[str, Any]:
        if self.local:
            return self.service.wie.inspect_file(source).to_dict()
        return self._request("POST", "/v1/weights/analyze", {"path": str(source)})

    def analyze(self, artifact_id: str) -> dict[str, Any]:
        if self.local:
            package = self.service.get_package(artifact_id)
            report = self.service.wie.inspect_package(package)
            return {"artifact_id": artifact_id, **report.to_dict()}
        return self._request("POST", "/v1/weights/analyze", {"artifact_id": artifact_id})

    def artifact(self, artifact_id: str) -> dict[str, Any]:
        if self.local:
            return self.service.artifact_manifest(artifact_id)
        return self._request("GET", f"/v1/artifacts/{artifact_id}")

    def verify_artifact(self, artifact_id: str) -> dict[str, Any]:
        if self.local:
            return self.service.verify_artifact(artifact_id)
        return self._request("GET", f"/v1/artifacts/{artifact_id}/verify")

    def lineage(self, artifact_id: str) -> dict[str, Any]:
        if self.local:
            return self.service.store.lineage_graph().describe(artifact_id)
        return self._request("GET", f"/v1/artifacts/{artifact_id}/lineage")

    def package(self, artifact_id: str) -> WeightPackage:
        if not self.local:
            raise UzenClientError(
                "raw packages are only available in local mode; use the REST/MCP interfaces "
                "to obtain artifacts as files"
            )
        return self.service.get_package(artifact_id)

    # ------------------------------------------------------------------ #
    # direct engineering calls (local mode executes immediately)
    # ------------------------------------------------------------------ #

    def modify(
        self,
        artifact_id: str,
        method: str,
        *,
        params: Mapping[str, Any] | None = None,
        others: Sequence[str] = (),
        seed: int | str | None = 0,
    ) -> dict[str, Any]:
        if not self.local:
            return self._request(
                "POST",
                "/v1/weights/modify",
                {"artifact_id": artifact_id, "method": method, "params": dict(params or {}), "others": list(others), "seed": seed},
            )
        from uzen.policies.strict_plus.effects import TaskScope

        service = self.service
        primary = service.get_package(artifact_id)
        other_pkgs = [service.get_package(a) for a in others]
        result = service.wme.modify(
            primary,
            method,
            params=params or {},
            others=other_pkgs,
            seed=seed,
            scope=TaskScope.WEIGHT_MODIFICATION,
        )
        record = service.store.put(
            result.package,
            operation=result.record.code,
            parents=[artifact_id] + list(others),
            operations=[result.record.to_dict()],
            rng_seeds={"main": result.record.seed},
            policy=result.record.policy,
            metadata={"method": method, "architecture_preserved": result.architecture_preserved},
        )
        service.db.index_artifact(
            artifact_id=record.artifact_id,
            path=str(record.path),
            name=result.package.name,
            family=result.package.family,
            operation=result.record.code,
            numel=result.package.numel(),
            num_tensors=len(result.package),
        )
        return {**result.to_dict(), "artifact_id": record.artifact_id}

    def generate(
        self,
        *,
        family: str | None = None,
        spec: Mapping[str, Any] | None = None,
        template_artifact_id: str | None = None,
        method: str = "orthogonal",
        params: Mapping[str, Any] | None = None,
        seed: int | str | None = 0,
    ) -> dict[str, Any]:
        if not self.local:
            return self._request(
                "POST",
                "/v1/weights/generate",
                {
                    "family": family,
                    "spec": dict(spec or {}),
                    "template_artifact_id": template_artifact_id,
                    "method": method,
                    "params": dict(params or {}),
                    "seed": seed,
                },
            )
        from uzen.core.vel.verify import verify
        from uzen.policies.strict_plus.effects import TaskScope

        service = self.service
        if family:
            result = service.wge.from_architecture(family, spec or {}, seed=seed)
            package = result.package
            operation = f"generate:{family}"
            definition = result.definition
        else:
            if not template_artifact_id:
                raise UzenClientError(
                    "generation needs either family= or template_artifact_id="
                )
            template = service.get_package(template_artifact_id)
            result = service.wge.generate(
                template, method=method, params=params or {}, seed=seed, scope=TaskScope.WEIGHT_GENERATION
            )
            package = result.package
            operation = result.record.code
            definition = None
        record = service.store.put(
            package,
            operation=operation,
            parents=[template_artifact_id] if template_artifact_id else [],
            operations=[result.record.to_dict()] if result.record else [],
            rng_seeds={"main": result.seed},
            metadata={
                "method": method,
                "architecture": definition.summary() if definition else None,
                "graph": definition.graph.to_dict() if definition else None,
            },
        )
        service.db.index_artifact(
            artifact_id=record.artifact_id,
            path=str(record.path),
            name=package.name,
            family=package.family,
            operation=operation,
            numel=package.numel(),
            num_tensors=len(package),
        )
        if definition is not None:
            service.db.register_model(
                artifact_id=record.artifact_id,
                family=package.family,
                spec={**definition.spec, "graph": definition.graph.to_dict()},
            )
            # Generation is only complete with evidence: bind the freshly built parameters
            # to the graph and run a real forward pass plus structural checks.
            evidence = verify(
                package,
                graph=definition.graph,
                inputs=definition.sample_inputs(seed=0),
            )
        else:
            evidence = verify(package, measure_performance=False)
        return {
            **result.to_dict(),
            "artifact_id": record.artifact_id,
            "verification": evidence.to_dict(),
        }

    def evaluate(
        self,
        artifact_id: str,
        *,
        reference_artifact_id: str | None = None,
        family: str | None = None,
        spec: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.local:
            return self._request(
                "POST",
                "/v1/weights/evaluate",
                {"artifact_id": artifact_id, "reference_artifact_id": reference_artifact_id,
                 "family": family, "spec": dict(spec or {})},
            )
        from uzen.core.vel.verify import verify

        service = self.service
        package = service.get_package(artifact_id)
        reference = service.get_package(reference_artifact_id) if reference_artifact_id else None
        graph = None
        inputs = None
        if family:
            from uzen.core.nws.families import build_architecture

            definition = build_architecture(family, spec or {})
            if definition.graph.validate(package).ok:
                graph = definition.graph
                inputs = definition.sample_inputs(seed=0)
        report = verify(package, reference=reference, graph=graph, inputs=inputs)
        return report.to_dict()

    # ------------------------------------------------------------------ #
    # jobs
    # ------------------------------------------------------------------ #

    def submit_job(self, request: Mapping[str, Any], *, idempotency_key: str | None = None) -> dict[str, Any]:
        if self.local:
            job = self.service.submit(request, idempotency_key=idempotency_key)
            return job
        return self._request("POST", "/v1/jobs", dict(request))

    def run_job(self, job_id: str) -> dict[str, Any]:
        if self.local:
            return self.service.run(job_id)
        return self._request("POST", f"/v1/jobs/{job_id}/run", {})

    def job(self, job_id: str) -> dict[str, Any]:
        if self.local:
            return self.service.get(job_id)
        return self._request("GET", f"/v1/jobs/{job_id}")

    def jobs(self, *, state: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if self.local:
            return self.service.list(state=state, limit=limit)
        query = f"?limit={limit}" + (f"&state={state}" if state else "")
        return self._request("GET", f"/v1/jobs{query}")["jobs"]

    def job_events(self, job_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        if self.local:
            return self.service.events(job_id, after=after)
        return self._request("GET", f"/v1/jobs/{job_id}/events?after={after}")["events"]

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        if self.local:
            return self.service.cancel(job_id)
        return self._request("POST", f"/v1/jobs/{job_id}/cancel", {})

    def run_local(self, request: Mapping[str, Any], *, idempotency_key: str | None = None) -> dict[str, Any]:
        """Submit and immediately execute a job (local mode only)."""
        if not self.local:
            raise UzenClientError("run_local is only available in local mode")
        job = self.submit_job(request, idempotency_key=idempotency_key)
        return self.run_job(job["id"])

    def stats(self) -> dict[str, Any]:
        if self.local:
            return self.service.stats()
        return self._request("GET", "/v1/stats")
