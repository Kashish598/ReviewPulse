"""Data drift detection for the reviews the API receives.

How it works
  * A reference snapshot (data/reference/reference.json) is built once from the training split by
    scripts/build_drift_reference.py. It stores, for a few simple features of a review, how the
    training data was spread over a handful of bins, plus the most common words (the vocabulary).
  * The API keeps a rolling window of the most recent requests, computes the same features, and
    compares the window to the reference with the Population Stability Index (PSI), per feature.
  * A feature drifts when its PSI reaches the threshold (0.25 by default, the usual rule of thumb
    for a significant shift). Any drifted feature means the window as a whole has drifted.
  * Nothing is decided until the window holds `min_window` requests (100 by default), because PSI
    is noisy on small samples.

Features: chars, words, non_ascii_ratio (other scripts, emoji), oov_rate (share of words outside
the training vocabulary) and positive (share of positive predictions, so a change in what the
model says is caught too). The first four are properties of the input, so no label is needed.

Use from the API: `drift.observe(text, pred_label)` after each prediction and
`drift.current_report()` for the /drift endpoint. Neither ever raises.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger("reviewpulse.drift")

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "data" / "reference" / "reference.json"
REFERENCE_VERSION = 1

WORD_RE = re.compile(r"\w+")
FEATURES = ("chars", "words", "non_ascii_ratio", "oov_rate", "positive")
# Features that are mostly zero or binary get fixed bin edges, because quantile bins would collapse.
FIXED_EDGES = {"non_ascii_ratio": [0.0, 0.02, 0.2], "positive": [0.5]}

DEFAULT_BINS = 5
DEFAULT_VOCAB_SIZE = 5000
DEFAULT_WINDOW_SIZE = 200
DEFAULT_MIN_WINDOW = 100
DEFAULT_PSI_THRESHOLD = 0.25
PSI_FLOOR = 1e-3  # empty bins count as this share, so one stray request cannot blow up the PSI


# ----------------------------------------------------------------------------- features


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def text_features(text: str, vocab: frozenset[str] | set[str]) -> dict[str, float]:
    words = tokenize(text)
    chars = len(text)
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    oov = sum(1 for w in words if w not in vocab) / len(words) if words else 1.0
    return {
        "chars": float(chars),
        "words": float(len(words)),
        "non_ascii_ratio": non_ascii / chars if chars else 0.0,
        "oov_rate": float(oov),
    }


def feature_row(text: str, pred_label: str, vocab: frozenset[str] | set[str]) -> dict[str, float]:
    row = text_features(text, vocab)
    row["positive"] = 1.0 if pred_label == "positive" else 0.0
    return row


# ----------------------------------------------------------------------------- statistics


def proportions(values: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    """Share of `values` in each bin. A value equal to an edge falls in the bin below it."""
    index = np.searchsorted(np.asarray(edges, dtype=float), np.asarray(values, dtype=float))
    counts = np.bincount(index, minlength=len(edges) + 1)
    return counts / max(len(values), 1)


def psi(reference: Sequence[float], current: Sequence[float]) -> float:
    ref = np.clip(np.asarray(reference, dtype=float), PSI_FLOOR, None)
    cur = np.clip(np.asarray(current, dtype=float), PSI_FLOOR, None)
    ref = ref / ref.sum()
    cur = cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def make_edges(name: str, values: Sequence[float], bins: int) -> list[float]:
    if name in FIXED_EDGES:
        return list(FIXED_EDGES[name])
    quantiles = np.quantile(np.asarray(values, dtype=float), np.linspace(0, 1, bins + 1)[1:-1])
    return [float(v) for v in np.unique(quantiles)]


# ----------------------------------------------------------------------------- reference


@dataclass(frozen=True)
class FeatureReference:
    edges: tuple[float, ...]
    proportions: tuple[float, ...]
    mean: float
    std: float


@dataclass(frozen=True)
class Reference:
    vocab: frozenset[str]
    features: dict[str, FeatureReference]
    meta: dict = field(default_factory=dict)


def build_reference(
    texts: Sequence[str],
    labels: Sequence,
    *,
    bins: int = DEFAULT_BINS,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    source: str = "",
) -> dict:
    """Build the reference snapshot (a JSON-ready dict) from reviews and their 0/1 labels.

    The true labels stand in for predictions in the `positive` feature: for a model that is right
    about 93% of the time the predicted mix is close to the true mix.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if len(texts) < 50:
        raise ValueError(f"Need at least 50 reviews to build a reference, got {len(texts)}")

    counts = Counter(word for text in texts for word in tokenize(text))
    vocab = [word for word, _ in counts.most_common(vocab_size)]
    vocab_set = set(vocab)

    positive_labels = {1, "1", "positive"}
    rows = [
        feature_row(text, "positive" if label in positive_labels else "negative", vocab_set)
        for text, label in zip(texts, labels, strict=True)
    ]

    features = {}
    for name in FEATURES:
        values = np.array([row[name] for row in rows], dtype=float)
        edges = make_edges(name, values, bins)
        features[name] = {
            "edges": edges,
            "proportions": [float(p) for p in proportions(values, edges)],
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    return {
        "version": REFERENCE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "n_reviews": len(texts),
        "bins": bins,
        "vocab_size": len(vocab),
        "vocab": vocab,
        "features": features,
    }


def save_reference(data: Mapping, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_reference(path: str | Path) -> Reference:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != REFERENCE_VERSION:
        raise ValueError(f"Unsupported drift reference version {data.get('version')!r}")
    missing = [name for name in FEATURES if name not in data.get("features", {})]
    if missing:
        raise ValueError(f"Drift reference is missing features: {missing}")
    features = {
        name: FeatureReference(
            edges=tuple(spec["edges"]),
            proportions=tuple(spec["proportions"]),
            mean=float(spec["mean"]),
            std=float(spec["std"]),
        )
        for name, spec in data["features"].items()
    }
    meta = {k: data.get(k) for k in ("created_at", "source", "n_reviews", "bins", "vocab_size")}
    return Reference(vocab=frozenset(data["vocab"]), features=features, meta=meta)


# ----------------------------------------------------------------------------- reports


@dataclass(frozen=True)
class FeatureDrift:
    psi: float
    drifted: bool
    reference_mean: float
    current_mean: float


@dataclass(frozen=True)
class DriftReport:
    status: str  # "ok", "drift", "insufficient_data" or "disabled"
    window: int
    min_window: int
    window_size: int
    psi_threshold: float
    checked_at: str
    features: dict[str, FeatureDrift] = field(default_factory=dict)

    @property
    def drifted_features(self) -> list[str]:
        return [name for name, item in self.features.items() if item.drifted]

    def to_dict(self) -> dict:
        """JSON-ready form, also what src/alerts.py formats into an email."""
        return {
            "status": self.status,
            "window": self.window,
            "min_window": self.min_window,
            "window_size": self.window_size,
            "psi_threshold": self.psi_threshold,
            "checked_at": self.checked_at,
            "drifted_features": self.drifted_features,
            "features": {
                name: {
                    "psi": round(item.psi, 4),
                    "drifted": item.drifted,
                    "reference_mean": round(item.reference_mean, 4),
                    "current_mean": round(item.current_mean, 4),
                }
                for name, item in self.features.items()
            },
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def evaluate_rows(
    reference: Reference,
    rows: Sequence[Mapping[str, float]],
    *,
    min_window: int = DEFAULT_MIN_WINDOW,
    window_size: int = DEFAULT_WINDOW_SIZE,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
) -> DriftReport:
    """Compare feature rows with the reference and say whether the data has drifted."""
    base = {
        "window": len(rows),
        "min_window": min_window,
        "window_size": window_size,
        "psi_threshold": psi_threshold,
        "checked_at": _now(),
    }
    if len(rows) < min_window:
        return DriftReport(status="insufficient_data", **base)

    features = {}
    for name in FEATURES:
        spec = reference.features[name]
        values = np.array([row[name] for row in rows], dtype=float)
        score = psi(spec.proportions, proportions(values, spec.edges))
        features[name] = FeatureDrift(
            psi=score,
            drifted=score >= psi_threshold,
            reference_mean=spec.mean,
            current_mean=float(values.mean()),
        )
    status = "drift" if any(item.drifted for item in features.values()) else "ok"
    return DriftReport(status=status, features=features, **base)


# ----------------------------------------------------------------------------- detector


class DriftDetector:
    """Rolling window of recent requests, compared with the reference on every request."""

    def __init__(
        self,
        reference: Reference,
        *,
        window_size: int = DEFAULT_WINDOW_SIZE,
        min_window: int = DEFAULT_MIN_WINDOW,
        psi_threshold: float = DEFAULT_PSI_THRESHOLD,
        on_drift: Callable[[DriftReport], None] | None = None,
    ):
        if min_window > window_size:
            raise ValueError("min_window cannot be larger than window_size")
        self.reference = reference
        self.window_size = window_size
        self.min_window = min_window
        self.psi_threshold = psi_threshold
        self.on_drift = on_drift
        self._rows: deque[dict[str, float]] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._last_status = "insufficient_data"

    def _evaluate(self) -> DriftReport:
        return evaluate_rows(
            self.reference,
            list(self._rows),
            min_window=self.min_window,
            window_size=self.window_size,
            psi_threshold=self.psi_threshold,
        )

    def observe(self, text: str, pred_label: str) -> DriftReport:
        row = feature_row(text, pred_label, self.reference.vocab)
        with self._lock:
            self._rows.append(row)
            report = self._evaluate()
            previous, self._last_status = self._last_status, report.status
        logger.debug("drift observe: row=%s window=%d status=%s", row, report.window, report.status)

        if report.status != previous and report.status in ("drift", "ok"):
            if report.status == "drift":
                worst = {n: round(report.features[n].psi, 3) for n in report.drifted_features}
                logger.warning("Data drift detected, PSI per drifted feature: %s", worst)
            elif previous == "drift":
                logger.info("Data drift cleared, the window matches the reference again")

        if report.status == "drift" and self.on_drift is not None:
            try:
                self.on_drift(report)
            except Exception:
                logger.exception("The drift callback failed, request handling is not affected")
        return report

    def report(self) -> DriftReport:
        with self._lock:
            return self._evaluate()

    def reset(self) -> None:
        with self._lock:
            self._rows.clear()
            self._last_status = "insufficient_data"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        on_drift: Callable[[DriftReport], None] | None = None,
    ) -> DriftDetector | None:
        """Build a detector from DRIFT_* variables, or None if there is no reference.

        DRIFT_REFERENCE, DRIFT_WINDOW_SIZE, DRIFT_MIN_WINDOW, DRIFT_PSI_THRESHOLD.
        """
        env = os.environ if env is None else env
        path = Path(env.get("DRIFT_REFERENCE") or REFERENCE_PATH)
        if not path.is_file():
            logger.warning(
                "Drift detection is off: no reference at %s. "
                "Run `python scripts/build_drift_reference.py` and commit the file.",
                path,
            )
            return None
        try:
            reference = load_reference(path)
        except (OSError, ValueError, KeyError) as error:
            logger.error("Drift detection is off: cannot read %s (%s)", path, error)
            return None

        window_size = _env_number(env, "DRIFT_WINDOW_SIZE", DEFAULT_WINDOW_SIZE, int)
        min_window = _env_number(env, "DRIFT_MIN_WINDOW", DEFAULT_MIN_WINDOW, int)
        threshold = _env_number(env, "DRIFT_PSI_THRESHOLD", DEFAULT_PSI_THRESHOLD, float)
        if min_window > window_size:
            logger.warning("DRIFT_MIN_WINDOW is above DRIFT_WINDOW_SIZE, using the window size")
            min_window = window_size
        logger.info(
            "Drift detection on: reference of %s reviews, window %d, minimum %d, PSI threshold %s",
            reference.meta.get("n_reviews"),
            window_size,
            min_window,
            threshold,
        )
        return cls(
            reference,
            window_size=window_size,
            min_window=min_window,
            psi_threshold=threshold,
            on_drift=on_drift,
        )


def _env_number(env: Mapping[str, str], name: str, default, cast):
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = cast(raw)
    except ValueError:
        logger.warning("%s=%r is not a number, using %s", name, raw, default)
        return default
    if value <= 0:
        logger.warning("%s must be positive, using %s", name, default)
        return default
    return value


# ----------------------------------------------------------------------------- module hooks

_state: dict = {"detector": None, "loaded": False, "on_drift": None}
_state_lock = threading.Lock()


def set_on_drift(callback: Callable[[DriftReport], None] | None) -> None:
    """Register what to call when the window has drifted (for example the email alert)."""
    with _state_lock:
        _state["on_drift"] = callback
        if _state["detector"] is not None:
            _state["detector"].on_drift = callback


def set_detector(detector: DriftDetector | None) -> None:
    """Replace the detector. Used by tests and by code that builds its own."""
    with _state_lock:
        _state["detector"] = detector
        _state["loaded"] = True


def reset_detector() -> None:
    """Forget the detector so the next call loads it again from the environment."""
    with _state_lock:
        _state.update(detector=None, loaded=False, on_drift=None)


def get_detector() -> DriftDetector | None:
    with _state_lock:
        if not _state["loaded"]:
            _state["detector"] = DriftDetector.from_env(on_drift=_state["on_drift"])
            _state["loaded"] = True
        return _state["detector"]


def observe(text: str, pred_label: str) -> DriftReport | None:
    """Record one request. Returns the report, or None when drift detection is off. Never raises."""
    try:
        detector = get_detector()
        return detector.observe(text, pred_label) if detector is not None else None
    except Exception:
        logger.exception("Drift observation failed, request handling is not affected")
        return None


def current_report() -> DriftReport:
    """The state for the /drift endpoint. Status is "disabled" when there is no reference."""
    detector = get_detector()
    if detector is None:
        return DriftReport(
            status="disabled",
            window=0,
            min_window=DEFAULT_MIN_WINDOW,
            window_size=DEFAULT_WINDOW_SIZE,
            psi_threshold=DEFAULT_PSI_THRESHOLD,
            checked_at=_now(),
        )
    return detector.report()
