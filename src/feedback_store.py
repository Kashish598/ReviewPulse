"""Saved API requests, the raw material for retraining.

Every review sent to /predict is saved with what the model said. A background labeller
(src/labeler.py) later asks an LLM for a label, and the labelled rows become new training data.

Privacy: the text people submit is stored and may be used to retrain the model. The README and
the /predict description say so. Nothing else about the caller (IP address, headers) is stored.

Database: DATABASE_URL. Neon or any Postgres works ("postgresql://..." is fine, it is rewritten
for the psycopg driver). With no DATABASE_URL a local SQLite file is used, which is lost when a
host without a persistent disk redeploys, so set DATABASE_URL in production. FEEDBACK_STORE=off
turns saving off.

Saving never breaks a request: use `record()` / `record_prediction()`, which swallow and log
errors. The other methods raise.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger("reviewpulse.store")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE_PATH = ROOT / "data" / "processed" / "feedback.db"  # git-ignored folder

MAX_TEXT_CHARS = 5000  # same limit as the API
VALIDATION_PERCENT = 20  # share of saved reviews kept aside as extra validation data
RAW_CHARS = 200  # how much of a rejected LLM answer is kept

PENDING = "pending"
LABELLED = "labelled"
DISCARDED = "discarded"
STATUSES = (PENDING, LABELLED, DISCARDED)

metadata = MetaData()

requests_table = Table(
    "requests",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("text", Text, nullable=False),
    Column("text_sha256", String(64), nullable=False, unique=True),
    Column("pred_label", String(8), nullable=False),
    Column("pred_score", Float),
    Column("model_name", String(100)),
    Column("model_version", String(100)),
    Column("split", String(12), nullable=False),  # "train" or "validation", fixed at save time
    Column("label_status", String(12), nullable=False, default=PENDING),
    Column("llm_label", String(8)),
    Column("llm_model", String(100)),
    Column("prompt_version", String(50)),
    Column("llm_raw", Text),  # the rejected answer, when the label was discarded
    Column("label_attempts", Integer, nullable=False, default=0),
    Column("last_attempt_at", DateTime(timezone=True)),
    Column("labelled_at", DateTime(timezone=True)),
    Index("ix_requests_status_id", "label_status", "id"),
)

label_usage_table = Table(
    "label_usage",
    metadata,
    Column("day", String(10), primary_key=True),  # UTC date, YYYY-MM-DD
    Column("calls", Integer, nullable=False, default=0),
    Column("tokens", Integer, nullable=False, default=0),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def assign_split(digest: str) -> str:
    """The same review always lands in the same split, so a review can never be in both."""
    return "validation" if int(digest[:8], 16) % 100 < VALIDATION_PERCENT else "train"


def normalise_database_url(url: str) -> str:
    """Neon and others give 'postgresql://' or 'postgres://'. Use the psycopg 3 driver for it."""
    url = url.strip()
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def redact_url(url: str) -> str:
    """The URL without its password, for logs."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    credentials, host = rest.rsplit("@", 1)
    user = credentials.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def make_engine(url: str) -> Engine:
    url = normalise_database_url(url)
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    # Neon suspends idle databases, so test a connection before using it and recycle old ones.
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,
        max_overflow=5,
        connect_args={"connect_timeout": 10},
    )


class FeedbackStore:
    def __init__(
        self,
        url: str | None = None,
        *,
        engine: Engine | None = None,
        clock: Callable[[], datetime] = utc_now,
    ):
        self._clock = clock
        if engine is not None:
            self.engine = engine
            self.description = "custom engine"
        else:
            if url is None:
                DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
                url = f"sqlite:///{DEFAULT_SQLITE_PATH}"
                logger.warning(
                    "No DATABASE_URL, saving requests to %s. That file is lost when a host "
                    "without a persistent disk redeploys, so set DATABASE_URL in production.",
                    DEFAULT_SQLITE_PATH,
                )
            self.engine = make_engine(url)
            self.description = redact_url(normalise_database_url(url))
        metadata.create_all(self.engine)
        logger.info("Feedback store ready: %s", self.description)

    @property
    def dialect(self) -> str:
        return self.engine.dialect.name

    # ------------------------------------------------------------ saving

    def add_request(
        self,
        text: str,
        pred_label: str,
        pred_score: float | None,
        model_name: str | None,
        model_version: str | None = None,
    ) -> int | None:
        """Save one request. Returns its id, or None when the same text is already saved."""
        text = text[:MAX_TEXT_CHARS]
        digest = text_hash(text)
        row = {
            "created_at": self._clock(),
            "text": text,
            "text_sha256": digest,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "model_name": model_name,
            "model_version": model_version,
            "split": assign_split(digest),
            "label_status": PENDING,
            "label_attempts": 0,
        }
        try:
            with self.engine.begin() as connection:
                new_id = connection.execute(
                    insert(requests_table).values(**row)
                ).inserted_primary_key[0]
        except IntegrityError:
            logger.debug("Request already saved (hash %s), skipping the duplicate", digest[:12])
            return None
        logger.debug("Saved request %s (%d chars, split %s)", new_id, len(text), row["split"])
        return int(new_id)

    def record(self, *args, **kwargs) -> int | None:
        """add_request that never raises, for use in the request path."""
        try:
            return self.add_request(*args, **kwargs)
        except Exception:
            logger.exception("Could not save the request, the prediction is not affected")
            return None

    # ------------------------------------------------------------ labelling

    def claim_pending(self, limit: int = 10) -> list[dict]:
        """The oldest requests still waiting for a label (those never tried come first)."""
        stmt = (
            select(
                requests_table.c.id,
                requests_table.c.text,
                requests_table.c.pred_label,
                requests_table.c.pred_score,
                requests_table.c.model_name,
            )
            .where(requests_table.c.label_status == PENDING)
            # Requests that already failed once go behind fresh ones, so one bad request
            # can never hold up everything behind it.
            .order_by(requests_table.c.label_attempts, requests_table.c.id)
            .limit(limit)
        )
        with self.engine.connect() as connection:
            return [dict(row._mapping) for row in connection.execute(stmt)]

    def _finish(self, request_id: int, **values) -> bool:
        stmt = (
            update(requests_table)
            .where(requests_table.c.id == request_id)
            .where(requests_table.c.label_status == PENDING)
            .values(labelled_at=self._clock(), last_attempt_at=self._clock(), **values)
        )
        with self.engine.begin() as connection:
            return connection.execute(stmt).rowcount == 1

    def mark_labelled(
        self, request_id: int, llm_label: str, llm_model: str, prompt_version: str
    ) -> bool:
        """Store the LLM's label. False when the request was already finished by someone else."""
        return self._finish(
            request_id,
            label_status=LABELLED,
            llm_label=llm_label,
            llm_model=llm_model,
            prompt_version=prompt_version,
            label_attempts=requests_table.c.label_attempts + 1,
        )

    def mark_discarded(
        self,
        request_id: int,
        reason: str,
        llm_model: str,
        prompt_version: str,
        count_attempt: bool = True,
    ) -> bool:
        """Give up on a request. Keeps what the LLM said. Set count_attempt=False when the
        failed attempt was already counted with record_attempt()."""
        values = {
            "label_status": DISCARDED,
            "llm_raw": reason[:RAW_CHARS],
            "llm_model": llm_model,
            "prompt_version": prompt_version,
        }
        if count_attempt:
            values["label_attempts"] = requests_table.c.label_attempts + 1
        return self._finish(request_id, **values)

    def record_attempt(self, request_id: int, note: str | None = None) -> int:
        """Count a failed labelling attempt and keep what the LLM said (first 200 characters).

        The request stays pending. Returns the new number of attempts, or 0 when the request
        is not pending any more.
        """
        stmt = (
            update(requests_table)
            .where(requests_table.c.id == request_id)
            .where(requests_table.c.label_status == PENDING)
            .values(
                label_attempts=requests_table.c.label_attempts + 1,
                last_attempt_at=self._clock(),
                llm_raw=(note or "")[:RAW_CHARS] or None,
            )
        )
        with self.engine.begin() as connection:
            if connection.execute(stmt).rowcount != 1:
                return 0
            attempts = select(requests_table.c.label_attempts).where(
                requests_table.c.id == request_id
            )
            return int(connection.execute(attempts).scalar() or 0)

    # ------------------------------------------------------------ daily usage

    def _today(self) -> str:
        return self._clock().astimezone(timezone.utc).strftime("%Y-%m-%d")

    def record_label_call(self, tokens: int = 0) -> int:
        """Count one LLM call against today's cap. Returns today's number of calls."""
        day = self._today()
        make_insert = pg_insert if self.dialect == "postgresql" else sqlite_insert
        statement = make_insert(label_usage_table).values(day=day, calls=1, tokens=tokens)
        statement = statement.on_conflict_do_update(
            index_elements=[label_usage_table.c.day],
            set_={
                "calls": label_usage_table.c.calls + 1,
                "tokens": label_usage_table.c.tokens + tokens,
            },
        )
        with self.engine.begin() as connection:
            connection.execute(statement)
        return self.calls_today()

    def calls_today(self) -> int:
        stmt = select(label_usage_table.c.calls).where(label_usage_table.c.day == self._today())
        with self.engine.connect() as connection:
            return int(connection.execute(stmt).scalar() or 0)

    def tokens_today(self) -> int:
        stmt = select(label_usage_table.c.tokens).where(label_usage_table.c.day == self._today())
        with self.engine.connect() as connection:
            return int(connection.execute(stmt).scalar() or 0)

    # ------------------------------------------------------------ reading

    def counts_by_status(self) -> dict[str, int]:
        stmt = select(requests_table.c.label_status, func.count()).group_by(
            requests_table.c.label_status
        )
        with self.engine.connect() as connection:
            found = {status: int(count) for status, count in connection.execute(stmt)}
        return {status: found.get(status, 0) for status in STATUSES}

    def labelled_rows(self, split: str | None = None) -> list[dict]:
        """Labelled requests (text and the LLM label), optionally only 'train' or 'validation'."""
        stmt = select(
            requests_table.c.id,
            requests_table.c.text,
            requests_table.c.llm_label,
            requests_table.c.pred_label,
            requests_table.c.split,
        ).where(requests_table.c.label_status == LABELLED)
        if split is not None:
            stmt = stmt.where(requests_table.c.split == split)
        with self.engine.connect() as connection:
            return [
                dict(row._mapping) for row in connection.execute(stmt.order_by(requests_table.c.id))
            ]

    def agreement_stats(self) -> dict:
        """How often the served model and the LLM gave the same label."""
        agree = (requests_table.c.pred_label == requests_table.c.llm_label).label("agree")
        stmt = select(agree, func.count()).where(requests_table.c.label_status == LABELLED)
        stmt = stmt.group_by(agree)
        with self.engine.connect() as connection:
            found = {bool(flag): int(count) for flag, count in connection.execute(stmt)}
        agree_n, disagree_n = found.get(True, 0), found.get(False, 0)
        total = agree_n + disagree_n
        return {
            "labelled": total,
            "agree": agree_n,
            "disagree": disagree_n,
            "agreement_rate": round(agree_n / total, 4) if total else None,
        }

    def dispose(self) -> None:
        self.engine.dispose()


# ----------------------------------------------------------------------------- module hooks

_state: dict = {"store": None, "loaded": False}
_state_lock = threading.Lock()


def get_store(env: Mapping[str, str] | None = None) -> FeedbackStore | None:
    """The shared store, created on first use. None when FEEDBACK_STORE=off or it cannot start."""
    with _state_lock:
        if _state["loaded"]:
            return _state["store"]
        _state["loaded"] = True
        env = os.environ if env is None else env
        if (env.get("FEEDBACK_STORE") or "").strip().lower() in {"off", "0", "false", "no"}:
            logger.info("Saving requests is turned off (FEEDBACK_STORE=off)")
            return None
        try:
            _state["store"] = FeedbackStore(env.get("DATABASE_URL") or None)
        except Exception:
            logger.exception("The feedback store could not start, requests will not be saved")
        return _state["store"]


def set_store(store: FeedbackStore | None) -> None:
    """Replace the shared store (tests), or pass None to load it again from the environment."""
    with _state_lock:
        _state["store"] = store
        _state["loaded"] = store is not None


def record_prediction(
    text: str,
    pred_label: str,
    pred_score: float | None,
    model_name: str | None,
    model_version: str | None = None,
) -> int | None:
    """Save one prediction. Never raises, so it is safe in a FastAPI background task."""
    try:
        store = get_store()
        if store is None:
            return None
        return store.record(text, pred_label, pred_score, model_name, model_version)
    except Exception:
        logger.exception("Could not save the request, the prediction is not affected")
        return None
