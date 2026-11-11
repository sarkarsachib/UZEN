"""Observability: metric registry rendering, JSON logs, and the /metrics endpoint."""

from __future__ import annotations

import json
import logging

import pytest

from uzen.infrastructure.observability.metrics import (
    JsonFormatter,
    Metrics,
    configure_logging,
    log_event,
)


def test_counters_gauges_and_labels_render_in_prometheus_format():
    metrics = Metrics()
    metrics.inc("uzen_jobs_total")
    metrics.inc("uzen_jobs_total")
    metrics.inc("uzen_errors_total", 2, code="policy.strict_plus")
    metrics.set("uzen_workers", 3)

    assert metrics.counter_value("uzen_jobs_total") == 2
    assert metrics.counter_value("uzen_errors_total", code="policy.strict_plus") == 2
    assert metrics.gauge_value("uzen_workers") == 3

    text = metrics.render()
    assert "# TYPE uzen_jobs_total counter" in text
    assert "uzen_jobs_total 2" in text
    assert 'uzen_errors_total{code="policy.strict_plus"} 2' in text
    assert "# TYPE uzen_workers gauge" in text
    assert "uzen_workers 3" in text


def test_histogram_buckets_are_cumulative_and_include_inf():
    metrics = Metrics(buckets=(0.1, 0.5, 1.0))
    metrics.observe("uzen_job_duration_seconds", 0.05, family="mlp")
    metrics.observe("uzen_job_duration_seconds", 0.3, family="mlp")
    metrics.observe("uzen_job_duration_seconds", 5.0, family="mlp")

    text = metrics.render()
    assert 'uzen_job_duration_seconds_bucket{family="mlp",le="0.1"} 1' in text
    assert 'uzen_job_duration_seconds_bucket{family="mlp",le="0.5"} 2' in text
    assert 'uzen_job_duration_seconds_bucket{family="mlp",le="1"} 2' in text
    assert 'uzen_job_duration_seconds_bucket{family="mlp",le="+Inf"} 3' in text
    assert 'uzen_job_duration_seconds_count{family="mlp"} 3' in text
    assert 'uzen_job_duration_seconds_sum{family="mlp"} 5.35' in text


def test_label_values_are_escaped():
    metrics = Metrics()
    metrics.inc("uzen_errors_total", code='quote"back\\slash\nnewline')
    text = metrics.render()
    assert 'code="quote\\"back\\\\slash\\nnewline"' in text


def test_timer_records_an_observation():
    metrics = Metrics()
    with metrics.timer("uzen_phase_seconds", phase="export"):
        pass
    values = metrics.histogram_values("uzen_phase_seconds", phase="export")
    assert len(values) == 1 and values[0] >= 0.0


def test_invalid_metric_names_are_rejected():
    with pytest.raises(ValueError):
        Metrics().inc("not a metric name")


def test_json_formatter_emits_one_parseable_object_with_fields():
    logger = logging.getLogger("uzen.test.observability")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.INFO)
    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    log_event(logger, "job.running", job_id="job-1", progress=0.5)

    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "job.running"
    assert payload["job_id"] == "job-1"
    assert payload["progress"] == 0.5
    assert payload["logger"] == "uzen.test.observability"


def test_configure_logging_installs_a_single_json_handler():
    logger = configure_logging("INFO", json_output=True)
    assert logger.name == "uzen"
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0].formatter, JsonFormatter)


def test_metrics_endpoint_serves_prometheus_text(tmp_path):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from uzen.interfaces.api.app import create_app

    app = create_app(root=tmp_path / "api-metrics")
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "# TYPE uzen_http_requests_total counter" in body
    assert 'path="/health"' in body
    assert "# TYPE uzen_jobs gauge" in body
