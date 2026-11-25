"""Unit tests for the timing counters behind /stats (src/monitoring.py)."""

import threading

from src.monitoring import RequestStats


def test_empty_snapshot_has_the_agreed_keys_and_no_averages():
    snapshot = RequestStats().snapshot()

    assert snapshot == {
        "request_count": 0,
        "avg_api_latency_ms": None,
        "avg_model_latency_ms": None,
        "per_model": {},
    }


def test_averages_and_per_model_block():
    stats = RequestStats()
    stats.record_request(10.0)
    stats.record_request(20.0)
    stats.record_model(
        "bert", 4.0, {"tokenize_ms": 1.0, "inference_ms": 2.5, "label_mapping_ms": 0.5}
    )
    stats.record_model(
        "bert", 6.0, {"tokenize_ms": 3.0, "inference_ms": 2.5, "label_mapping_ms": 0.5}
    )

    snapshot = stats.snapshot()

    assert snapshot["request_count"] == 2
    assert snapshot["avg_api_latency_ms"] == 15.0
    assert snapshot["avg_model_latency_ms"] == 5.0
    assert snapshot["per_model"]["bert"] == {
        "count": 2,
        "avg_model_ms": 5.0,
        "avg_tokenize_ms": 2.0,
        "avg_inference_ms": 2.5,
        "avg_label_mapping_ms": 0.5,
    }


def test_a_model_without_stage_times_shows_none_for_the_stages():
    stats = RequestStats()
    stats.record_model("mock", 1.0, {"total_ms": 1.0})

    block = stats.snapshot()["per_model"]["mock"]

    assert block["count"] == 1
    assert block["avg_model_ms"] == 1.0
    assert block["avg_tokenize_ms"] is None
    assert block["avg_inference_ms"] is None
    assert block["avg_label_mapping_ms"] is None


def test_models_are_counted_separately():
    stats = RequestStats()
    stats.record_model("a", 2.0)
    stats.record_model("b", 4.0)

    snapshot = stats.snapshot()

    assert snapshot["per_model"]["a"]["count"] == 1
    assert snapshot["per_model"]["b"]["avg_model_ms"] == 4.0
    assert snapshot["avg_model_latency_ms"] == 3.0


def test_reset_clears_everything():
    stats = RequestStats()
    stats.record_request(5.0)
    stats.record_model("a", 2.0)

    stats.reset()

    assert stats.snapshot()["request_count"] == 0
    assert stats.snapshot()["per_model"] == {}


def test_counting_is_safe_across_threads():
    stats = RequestStats()

    def work():
        for _ in range(500):
            stats.record_request(1.0)
            stats.record_model("m", 1.0)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    snapshot = stats.snapshot()
    assert snapshot["request_count"] == 4000
    assert snapshot["per_model"]["m"]["count"] == 4000
