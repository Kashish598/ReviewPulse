"""Drift detector + alert + simulator working together. Needs src/drift.py, src/alerts.py and
scripts/simulate_drift.py, so add it only after the drift and alerts branches are merged.
"""

import importlib.util
import random
import sys
from pathlib import Path

import pytest

from src import alerts, drift

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "simulate_drift", ROOT / "scripts" / "simulate_drift.py"
)
sim = importlib.util.module_from_spec(_spec)
sys.modules["simulate_drift"] = sim
_spec.loader.exec_module(sim)


class Recorder:
    def __init__(self):
        self.sent = []

    def __call__(self, smtp_settings, subject, body):
        self.sent.append((subject, body))


@pytest.fixture
def reference():
    rng = random.Random(0)
    texts = [rng.choice(sim.NORMAL_REVIEWS) for _ in range(800)]  # same source as normal traffic
    data = drift.build_reference(texts, [rng.randint(0, 1) for _ in texts], vocab_size=100)
    return drift.Reference(
        vocab=frozenset(data["vocab"]),
        features={
            name: drift.FeatureReference(
                edges=tuple(spec["edges"]),
                proportions=tuple(spec["proportions"]),
                mean=spec["mean"],
                std=spec["std"],
            )
            for name, spec in data["features"].items()
        },
    )


@pytest.fixture
def wired(reference):
    recorder = Recorder()
    alerts.set_manager(
        alerts.AlertManager(
            alerts.SmtpSettings("smtp.example.com", 587, "u", "p", "u@x.com", ("to@x.com",)),
            sender=recorder,
            threaded=False,
        )
    )
    detector = drift.DriftDetector(reference, on_drift=lambda r: alerts.alert_drift(r.to_dict()))
    yield detector, recorder
    alerts.set_manager(None)


def send(detector, texts):
    report = None
    for index, text in enumerate(texts):
        report = detector.observe(text, "positive" if index % 2 else "negative")
    return report


def test_normal_traffic_sends_no_email(wired):
    detector, recorder = wired

    report = send(detector, sim.build_traffic("normal", 200, seed=1, csv_path=Path("none.csv")))

    assert report.status == "ok"
    assert recorder.sent == []


def test_shifted_traffic_sends_one_email_that_names_what_changed(wired):
    detector, recorder = wired

    report = send(detector, sim.build_traffic("shifted", 150, seed=2))

    assert report.status == "drift"
    assert len(recorder.sent) == 1  # drift persists over 50 requests, the cooldown allows one email
    subject, body = recorder.sent[0]
    assert subject.startswith("[ReviewPulse] Data drift detected")
    assert "non_ascii_ratio" in body
