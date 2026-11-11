"""Dependency-free metrics and structured logging for UZEN.

Counters, gauges and histograms with Prometheus text exposition, plus JSON log
formatting. Everything here is local to the process; a deployment forwards ``/metrics``
to Prometheus (or any OpenMetrics scraper) and ships JSON logs to its log pipeline.

The process-wide registry is :data:`METRICS`. Services can also construct an isolated
:class:`Metrics` instance for tests.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, MutableMapping

DEFAULT_BUCKETS: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

_METRIC_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:")


def _check_name(name: str) -> str:
    if not name or any(c not in _METRIC_NAME_CHARS for c in name):
        raise ValueError(f"invalid metric name {name!r}; use [A-Za-z0-9_:]")
    return name


def _format_labels(labels: MutableMapping[str, Any]) -> str:
    if not labels:
        return ""
    parts = []
    for key in sorted(labels):
        value = str(labels[key]).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
        parts.append(f'{key}="{value}"')
    return "{" + ",".join(parts) + "}"


class Metrics:
    """Thread-safe in-process metric registry."""

    def __init__(self, *, buckets: tuple[float, ...] = DEFAULT_BUCKETS) -> None:
        self.buckets = tuple(sorted(float(b) for b in buckets))
        self._lock = threading.RLock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[
            tuple[str, tuple[tuple[str, str], ...]], list[float]
        ] = {}

    @staticmethod
    def _key(name: str, labels: MutableMapping[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
        return _check_name(name), tuple(sorted((str(k), str(v)) for k, v in labels.items()))

    # ------------------------------------------------------------------ #

    def inc(self, name: str, value: float = 1.0, /, **labels: Any) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + float(value)

    def set(self, name: str, value: float, /, **labels: Any) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._gauges[key] = float(value)

    def observe(self, name: str, value: float, /, **labels: Any) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._histograms.setdefault(key, []).append(float(value))

    def counter_value(self, name: str, **labels: Any) -> float:
        key = self._key(name, labels)
        with self._lock:
            return self._counters.get(key, 0.0)

    def gauge_value(self, name: str, **labels: Any) -> float:
        key = self._key(name, labels)
        with self._lock:
            return self._gauges.get(key, 0.0)

    def histogram_values(self, name: str, **labels: Any) -> list[float]:
        key = self._key(name, labels)
        with self._lock:
            return list(self._histograms.get(key, ()))

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    @contextmanager
    def timer(self, name: str, /, **labels: Any) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - started, **labels)

    # ------------------------------------------------------------------ #

    def render(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = {k: list(v) for k, v in self._histograms.items()}

        for (name, labels), value in sorted(counters.items()):
            lines.extend(
                [
                    f"# TYPE {name} counter",
                    f"{name}{_format_labels(dict(labels))} {_format_number(value)}",
                ]
            )
        for (name, labels), value in sorted(gauges.items()):
            lines.extend(
                [
                    f"# TYPE {name} gauge",
                    f"{name}{_format_labels(dict(labels))} {_format_number(value)}",
                ]
            )
        for (name, labels), values in sorted(histograms.items()):
            base = dict(labels)
            lines.append(f"# TYPE {name} histogram")
            for bound in self.buckets:
                cumulative = sum(1 for v in values if v <= bound)
                bucket_labels = dict(base)
                bucket_labels["le"] = _format_number(bound)
                lines.append(f"{name}_bucket{_format_labels(bucket_labels)} {cumulative}")
            bucket_labels = dict(base)
            bucket_labels["le"] = "+Inf"
            lines.append(f"{name}_bucket{_format_labels(bucket_labels)} {len(values)}")
            lines.append(f"{name}_sum{_format_labels(base)} {_format_number(sum(values))}")
            lines.append(f"{name}_count{_format_labels(base)} {len(values)}")
        return "\n".join(lines) + ("\n" if lines else "")


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


METRICS = Metrics()


# --------------------------------------------------------------------------- #
# structured logging
# --------------------------------------------------------------------------- #


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with optional ``extra={"fields": {...}}``."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> logging.Logger:
    """Install a single stderr handler on the ``uzen`` logger and return it."""
    logger = logging.getLogger("uzen")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter() if json_output else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, /, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": {"event": event, **fields}})
