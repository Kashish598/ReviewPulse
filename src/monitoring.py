# src/monitoring.py and src/app.py (K2, K3).

# RequestStats holds thread-safe counters. A middleware in app.py times each /predict request,
# and the model records its own time per stage (tokenize, inference, label mapping).
# /stats and /metrics return the same JSON: request_count, avg_api_latency_ms,
# avg_model_latency_ms and per_model. Only successful /predict calls are counted.
# Counters reset on restart.
# The API time is always larger than the model time, because it includes validation
# and the response.
# In app.py, /health is a liveness check that never loads the model. /ready returns 200 once
# the model has loaded, and 503 while loading or on failure. The model loads at startup,
# which is why your log said "Model ready after 4.4 s".
# Tests: tests/test_monitoring.py and tests/test_app_health_stats.py.

"""Request and model timing counters behind /stats and /metrics (K2).

Only successful /predict calls are counted, so the API average and the model average describe the
same requests and can be compared (the API time includes the model time plus validation, saving
the response and so on). Everything is kept in memory and is lost on restart.
"""

from __future__ import annotations

import threading

STAGES = ("tokenize_ms", "inference_ms", "label_mapping_ms")


class RequestStats:
    """Thread-safe counters. The API handles requests in several threads at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count = 0
        self._api_ms_total = 0.0
        self._models: dict[str, dict] = {}

    def reset(self) -> None:
        with self._lock:
            self._request_count = 0
            self._api_ms_total = 0.0
            self._models = {}

    def record_request(self, api_ms: float) -> None:
        """One finished /predict request, timed from the middleware."""
        with self._lock:
            self._request_count += 1
            self._api_ms_total += api_ms

    def record_model(self, model_name: str, model_ms: float, stages: dict | None = None) -> None:
        """One model call. `stages` holds tokenize_ms, inference_ms and label_mapping_ms when the
        model reports them (the BERT adapter does, the mock does not)."""
        with self._lock:
            entry = self._models.setdefault(
                model_name,
                {"count": 0, "model_ms": 0.0, "stages": {s: [0.0, 0] for s in STAGES}},
            )
            entry["count"] += 1
            entry["model_ms"] += model_ms
            for stage in STAGES:
                if stages and stage in stages:
                    entry["stages"][stage][0] += float(stages[stage])
                    entry["stages"][stage][1] += 1

    def snapshot(self) -> dict:
        """The JSON returned by /stats and /metrics (keys agreed in the M2 plan, Section 5)."""
        with self._lock:
            model_calls = sum(m["count"] for m in self._models.values())
            model_total = sum(m["model_ms"] for m in self._models.values())
            per_model = {}
            for name, entry in self._models.items():
                block = {
                    "count": entry["count"],
                    "avg_model_ms": _avg(entry["model_ms"], entry["count"]),
                }
                for stage, (total, n) in entry["stages"].items():
                    block[f"avg_{stage}"] = _avg(total, n)  # None if the model never reported it
                per_model[name] = block
            return {
                "request_count": self._request_count,
                "avg_api_latency_ms": _avg(self._api_ms_total, self._request_count),
                "avg_model_latency_ms": _avg(model_total, model_calls),
                "per_model": per_model,
            }


def _avg(total: float, n: int) -> float | None:
    return round(total / n, 3) if n else None


stats = RequestStats()
