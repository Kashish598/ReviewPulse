import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "simulate_drift.py"
_spec = importlib.util.spec_from_file_location("simulate_drift", SCRIPT)
sim = importlib.util.module_from_spec(_spec)
sys.modules["simulate_drift"] = sim
_spec.loader.exec_module(sim)


@pytest.fixture
def server():
    received = {"predict": [], "drift_calls": 0}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            received["predict"].append(json.loads(self.rfile.read(length))["text"])
            self._send(200, {"sentiment": "positive", "score": 0.9})

        def do_GET(self):
            if self.path == "/drift":
                received["drift_calls"] += 1
                self._send(200, {"status": "drift", "window": 10, "drifted_features": ["chars"]})
            else:
                self._send(404, {"detail": "not found"})

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", received
    httpd.shutdown()
    httpd.server_close()


def test_shifted_traffic_is_valid_for_the_api_and_looks_different():
    texts = sim.build_traffic("shifted", 40, seed=1)

    assert len(texts) == 40
    assert all(t.strip() and len(t) <= 5000 for t in texts)  # the API rejects blank or > 5000
    assert any(len(t) > 3000 for t in texts)
    assert any(not t.isascii() for t in texts)
    assert any(len(t) < 10 for t in texts)


def test_traffic_is_reproducible_for_a_seed():
    assert sim.build_traffic("mixed", 30, seed=5) == sim.build_traffic("mixed", 30, seed=5)
    assert sim.build_traffic("mixed", 30, seed=5) != sim.build_traffic("mixed", 30, seed=6)


def test_normal_traffic_is_ordinary_english_and_mixed_has_both(tmp_path):
    missing_csv = tmp_path / "none.csv"

    normal = sim.build_traffic("normal", 30, csv_path=missing_csv)
    mixed = sim.build_traffic("mixed", 30, csv_path=missing_csv)

    assert all(t.isascii() for t in normal)
    assert any(not t.isascii() for t in mixed) and any(t in sim.NORMAL_REVIEWS for t in mixed)


def test_normal_traffic_can_come_from_a_csv(tmp_path):
    path = tmp_path / "reviews.csv"
    pd.DataFrame({"review_text": ["only this one", "  "], "label": [1, 0]}).to_csv(
        path, index=False
    )

    assert set(sim.build_traffic("normal", 5, csv_path=path)) == {"only this one"}


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        sim.build_traffic("loud", 5)


def test_main_sends_every_request_and_reads_the_drift_verdict(server):
    url, received = server

    code = sim.main(["--url", url, "--mode", "mixed", "--count", "12", "--delay", "0"])

    assert code == 0
    assert len(received["predict"]) == 12
    assert received["drift_calls"] == 1


def test_main_fails_when_nothing_can_be_reached():
    assert sim.main(["--url", "http://127.0.0.1:9", "--count", "2", "--delay", "0"]) == 1


def test_main_copes_with_an_api_that_has_no_drift_endpoint(server, monkeypatch):
    url, received = server
    original = sim.request_json

    def without_drift(target, payload=None, timeout=30.0):
        if target.endswith("/drift"):
            return original(target.replace("/drift", "/missing"), payload, timeout)
        return original(target, payload, timeout)

    monkeypatch.setattr(sim, "request_json", without_drift)

    assert sim.main(["--url", url, "--count", "3", "--delay", "0"]) == 0
