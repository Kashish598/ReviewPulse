"""Email alerts, used for data drift.

    SMTP_USER, SMTP_PASSWORD   the sending account (for Gmail, an app password, not the login)
    ALERT_TO                   who is told, comma separated
    SMTP_HOST, SMTP_PORT       default smtp.gmail.com and 587 (STARTTLS); port 465 uses SSL
    ALERT_FROM                 default SMTP_USER
    ALERT_COOLDOWN_SECONDS     default 3600, at most one email per alert key in that time

Every alert is written to the log first, so it exists even when email is not set up. Sending
happens in a background thread and never raises, so a mail problem cannot break a request.
Secrets come from the environment (platform secrets on Render), never from the repository.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage

logger = logging.getLogger("reviewpulse.alerts")

DEFAULT_HOST = "smtp.gmail.com"
DEFAULT_PORT = 587
DEFAULT_COOLDOWN_SECONDS = 3600.0
RETRY_AFTER_FAILURE_SECONDS = 60.0


@dataclass(frozen=True)
class SmtpSettings:
    host: str
    port: int
    username: str
    password: str = field(repr=False)  # repr=False keeps it out of logs and tracebacks
    sender: str = ""
    recipients: tuple[str, ...] = ()
    use_ssl: bool = False
    timeout: float = 15.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SmtpSettings | None:
        """Settings from the environment, or None when email is not configured."""
        env = os.environ if env is None else env
        username = (env.get("SMTP_USER") or "").strip()
        password = env.get("SMTP_PASSWORD") or ""
        recipients = tuple(
            item.strip()
            for item in (env.get("ALERT_TO") or "").replace(";", ",").split(",")
            if item.strip()
        )
        missing = [
            name
            for name, value in (
                ("SMTP_USER", username),
                ("SMTP_PASSWORD", password),
                ("ALERT_TO", recipients),
            )
            if not value
        ]
        if missing:
            logger.debug("Email alerts are off, not set: %s", ", ".join(missing))
            return None

        try:
            port = int(env.get("SMTP_PORT") or DEFAULT_PORT)
        except ValueError:
            logger.warning("SMTP_PORT is not a number, using %d", DEFAULT_PORT)
            port = DEFAULT_PORT
        return cls(
            host=(env.get("SMTP_HOST") or DEFAULT_HOST).strip(),
            port=port,
            username=username,
            password=password,
            sender=(env.get("ALERT_FROM") or username).strip(),
            recipients=recipients,
            use_ssl=port == 465,
        )


def clean_subject(subject: str) -> str:
    """One line, no header injection, not absurdly long."""
    return " ".join(subject.split())[:200]


def send_email(settings: SmtpSettings, subject: str, body: str) -> None:
    """Send one plain-text email. Raises on any failure, the caller decides what to do."""
    message = EmailMessage()
    message["Subject"] = clean_subject(subject)
    message["From"] = settings.sender
    message["To"] = ", ".join(settings.recipients)
    message.set_content(body)

    if settings.use_ssl:
        with smtplib.SMTP_SSL(
            settings.host,
            settings.port,
            timeout=settings.timeout,
            context=ssl.create_default_context(),
        ) as smtp:
            smtp.login(settings.username, settings.password)
            smtp.send_message(message)
        return

    with smtplib.SMTP(settings.host, settings.port, timeout=settings.timeout) as smtp:
        smtp.ehlo()
        smtp.starttls(context=ssl.create_default_context())
        smtp.ehlo()
        smtp.login(settings.username, settings.password)
        smtp.send_message(message)


class AlertManager:
    """Sends an alert at most once per key per cooldown, without blocking the caller."""

    def __init__(
        self,
        settings: SmtpSettings | None,
        *,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        retry_seconds: float = RETRY_AFTER_FAILURE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sender: Callable[[SmtpSettings, str, str], None] = send_email,
        threaded: bool = True,
    ):
        self.settings = settings
        self.cooldown_seconds = cooldown_seconds
        self.retry_seconds = retry_seconds
        self._clock = clock
        self._sender = sender
        self._threaded = threaded
        self._next_allowed: dict[str, float] = {}
        self._lock = threading.Lock()
        self._warned_unconfigured = False

    def notify(self, key: str, subject: str, body: str) -> bool:
        """Log the alert and email it unless this key is cooling down. True if a send was tried."""
        clean = clean_subject(subject)
        logger.warning("ALERT [%s] %s", key, clean)

        if self.settings is None:
            if not self._warned_unconfigured:
                self._warned_unconfigured = True
                logger.warning(
                    "No email sent, set SMTP_USER, SMTP_PASSWORD and ALERT_TO to enable them"
                )
            return False

        with self._lock:
            now = self._clock()
            if now < self._next_allowed.get(key, 0.0):
                logger.debug("Alert [%s] is cooling down, no email sent", key)
                return False
            # Reserve the slot now so two requests at once cannot send two emails.
            self._next_allowed[key] = now + self.cooldown_seconds

        if self._threaded:
            threading.Thread(
                target=self._deliver, args=(key, clean, body), name=f"alert-{key}", daemon=True
            ).start()
        else:
            self._deliver(key, clean, body)
        return True

    def _deliver(self, key: str, subject: str, body: str) -> None:
        try:
            self._sender(self.settings, subject, body)
        except Exception:
            logger.exception("Could not send the alert email [%s], will retry later", key)
            with self._lock:
                self._next_allowed[key] = self._clock() + self.retry_seconds
        else:
            logger.info("Alert email [%s] sent to %s", key, ", ".join(self.settings.recipients))


def format_drift_message(report: Mapping) -> tuple[str, str]:
    """Subject and body for a drift report (the dict from DriftReport.to_dict())."""
    drifted = list(report.get("drifted_features", []))
    features = report.get("features", {})
    threshold = report.get("psi_threshold")

    subject = f"[ReviewPulse] Data drift detected in {len(drifted)} feature(s): " + ", ".join(
        drifted
    )

    def line(name: str) -> str:
        item = features.get(name, {})
        return (
            f"  {name:<16} PSI {item.get('psi', 0):<8} "
            f"reference mean {item.get('reference_mean', 0):<10} "
            f"current mean {item.get('current_mean', 0)}"
        )

    others = [name for name in features if name not in drifted]
    lines = [
        "ReviewPulse is receiving data that looks different from its training data.",
        "",
        f"Time (UTC):            {report.get('checked_at') or datetime.now(timezone.utc)}",
        f"Requests in window:    {report.get('window')} (needs {report.get('min_window')})",
        f"PSI threshold:         {threshold} (above 0.25 is the usual sign of a real shift)",
        "",
        "Drifted features:",
        *[line(name) for name in drifted],
    ]
    if others:
        lines += ["", "Features still in range:", *[line(name) for name in others]]
    lines += [
        "",
        "What the features mean:",
        "  chars, words      length of the review",
        "  non_ascii_ratio   share of characters outside plain English (other scripts, emoji)",
        "  oov_rate          share of words the training data never used often",
        "  positive          share of reviews the model called positive",
        "",
        "Next steps: open GET /drift for the live numbers, check logs/reviewpulse.log, and look at",
        "the recent requests. If the shift is real, retrain on newer data.",
        "",
        "This email is sent at most once per cooldown period while the drift lasts.",
    ]
    return subject, "\n".join(lines)


# ---------------------------------------------------------------------------- module hook

_manager: AlertManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> AlertManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            try:
                cooldown = float(
                    os.environ.get("ALERT_COOLDOWN_SECONDS") or DEFAULT_COOLDOWN_SECONDS
                )
            except ValueError:
                logger.warning("ALERT_COOLDOWN_SECONDS is not a number, using the default")
                cooldown = DEFAULT_COOLDOWN_SECONDS
            _manager = AlertManager(SmtpSettings.from_env(), cooldown_seconds=cooldown)
        return _manager


def set_manager(manager: AlertManager | None) -> None:
    """Replace the shared manager (tests), or pass None to rebuild it from the environment."""
    global _manager
    with _manager_lock:
        _manager = manager


def alert_drift(report: Mapping) -> bool:
    """Alert on a drift report dict. Never raises, so it is safe to use as the drift callback."""
    try:
        subject, body = format_drift_message(report)
        return get_manager().notify("data-drift", subject, body)
    except Exception:
        logger.exception("Drift alert failed")
        return False
