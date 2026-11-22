import logging
import threading

import pytest

from src import alerts
from src.alerts import AlertManager, SmtpSettings

ENV = {
    "SMTP_USER": "sender@example.com",
    "SMTP_PASSWORD": "app-password",
    "ALERT_TO": "a@example.com; b@example.com, ,c@example.com",
}

DRIFT_REPORT = {
    "status": "drift",
    "window": 120,
    "min_window": 100,
    "window_size": 200,
    "psi_threshold": 0.25,
    "checked_at": "2026-11-25T10:00:00+00:00",
    "drifted_features": ["oov_rate", "non_ascii_ratio"],
    "features": {
        "chars": {"psi": 0.02, "drifted": False, "reference_mean": 380.0, "current_mean": 395.0},
        "oov_rate": {"psi": 0.9, "drifted": True, "reference_mean": 0.12, "current_mean": 0.7},
        "non_ascii_ratio": {
            "psi": 0.6,
            "drifted": True,
            "reference_mean": 0.0,
            "current_mean": 0.4,
        },
    },
}


class FakeSMTP:
    """Records what the code does with smtplib, never touches the network."""

    instances: list = []

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.timeout, self.context = host, port, timeout, context
        self.calls = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def ehlo(self):
        self.calls.append("ehlo")

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, message):
        self.calls.append(("send", message))


@pytest.fixture(autouse=True)
def reset_state():
    FakeSMTP.instances = []
    alerts.set_manager(None)
    yield
    alerts.set_manager(None)


def settings(**overrides):
    base = dict(
        host="smtp.example.com",
        port=587,
        username="u",
        password="p",
        sender="u@example.com",
        recipients=("to@example.com",),
    )
    return SmtpSettings(**{**base, **overrides})


class Recorder:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def __call__(self, smtp_settings, subject, body):
        if self.fail:
            raise OSError("connection refused")
        self.sent.append((subject, body))


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# ------------------------------------------------------------------ settings


def test_settings_from_env_parse_recipients_and_defaults():
    result = SmtpSettings.from_env(ENV)

    assert result.recipients == ("a@example.com", "b@example.com", "c@example.com")
    assert (result.host, result.port, result.use_ssl) == ("smtp.gmail.com", 587, False)
    assert result.sender == "sender@example.com"


def test_port_465_means_ssl_and_a_bad_port_falls_back():
    assert SmtpSettings.from_env({**ENV, "SMTP_PORT": "465"}).use_ssl is True
    assert SmtpSettings.from_env({**ENV, "SMTP_PORT": "abc"}).port == 587


@pytest.mark.parametrize("missing", ["SMTP_USER", "SMTP_PASSWORD", "ALERT_TO"])
def test_settings_are_none_when_something_required_is_missing(missing):
    env = {k: v for k, v in ENV.items() if k != missing}

    assert SmtpSettings.from_env(env) is None


def test_password_is_not_shown_in_repr():
    assert "app-password" not in repr(SmtpSettings.from_env(ENV))


# ------------------------------------------------------------------ sending


def test_send_email_uses_starttls_login_and_a_clean_message(monkeypatch):
    monkeypatch.setattr(alerts.smtplib, "SMTP", FakeSMTP)

    alerts.send_email(settings(), "Subject\nBcc: evil@example.com", "the body")

    smtp = FakeSMTP.instances[0]
    assert (smtp.host, smtp.port) == ("smtp.example.com", 587)
    assert smtp.calls[:3] == ["ehlo", "starttls", "ehlo"]
    assert smtp.calls[3] == ("login", "u", "p")
    message = smtp.calls[4][1]
    assert message["Subject"] == "Subject Bcc: evil@example.com"
    assert message["Bcc"] is None
    assert message["To"] == "to@example.com"
    assert "the body" in message.get_content()


def test_send_email_uses_ssl_on_port_465(monkeypatch):
    monkeypatch.setattr(alerts.smtplib, "SMTP_SSL", FakeSMTP)

    alerts.send_email(settings(port=465, use_ssl=True), "s", "b")

    smtp = FakeSMTP.instances[0]
    assert "starttls" not in smtp.calls
    assert smtp.calls[0] == ("login", "u", "p")


# ------------------------------------------------------------------ cooldown


def test_first_alert_is_sent_and_repeats_are_held_back_until_the_cooldown_ends():
    recorder, clock = Recorder(), Clock()
    manager = AlertManager(
        settings(), cooldown_seconds=3600, clock=clock, sender=recorder, threaded=False
    )

    assert manager.notify("drift", "s", "b") is True
    clock.now += 100
    assert manager.notify("drift", "s", "b") is False
    clock.now += 3600
    assert manager.notify("drift", "s", "b") is True

    assert len(recorder.sent) == 2


def test_different_keys_have_separate_cooldowns():
    recorder = Recorder()
    manager = AlertManager(settings(), sender=recorder, clock=Clock(), threaded=False)

    assert manager.notify("drift", "s", "b") is True
    assert manager.notify("other", "s", "b") is True


def test_a_failed_send_is_contained_and_retried_after_a_short_wait(caplog):
    clock = Clock()
    broken = Recorder(fail=True)
    manager = AlertManager(
        settings(),
        cooldown_seconds=3600,
        retry_seconds=60,
        clock=clock,
        sender=broken,
        threaded=False,
    )

    with caplog.at_level(logging.ERROR, logger="reviewpulse.alerts"):
        assert manager.notify("drift", "s", "b") is True

    assert any("Could not send" in r.getMessage() for r in caplog.records)
    clock.now += 30
    assert manager.notify("drift", "s", "b") is False
    clock.now += 31
    manager._sender = Recorder()
    assert manager.notify("drift", "s", "b") is True


def test_without_email_settings_the_alert_is_still_logged(caplog):
    manager = AlertManager(None, threaded=False)

    with caplog.at_level(logging.WARNING, logger="reviewpulse.alerts"):
        assert manager.notify("drift", "Data drift detected", "body") is False
        manager.notify("drift", "Data drift detected", "body")

    messages = [r.getMessage() for r in caplog.records]
    assert sum("ALERT [drift] Data drift detected" in m for m in messages) == 2
    assert sum("No email sent" in m for m in messages) == 1  # explained once, not every time


def test_notify_does_not_block_the_caller():
    started, release, done = threading.Event(), threading.Event(), threading.Event()

    def slow_sender(smtp_settings, subject, body):
        started.set()
        release.wait(5)
        done.set()

    manager = AlertManager(settings(), sender=slow_sender, threaded=True)

    assert manager.notify("drift", "s", "b") is True  # returns while the sender is still blocked
    assert started.wait(5)
    assert not done.is_set()
    release.set()
    assert done.wait(5)


def test_two_simultaneous_alerts_send_one_email():
    recorder = Recorder()
    manager = AlertManager(settings(), sender=recorder, clock=Clock(), threaded=False)
    results = []

    threads = [
        threading.Thread(target=lambda: results.append(manager.notify("drift", "s", "b")))
        for _ in range(10)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(True) == 1
    assert len(recorder.sent) == 1


# ------------------------------------------------------------------ message


def test_drift_message_names_the_features_and_the_numbers():
    subject, body = alerts.format_drift_message(DRIFT_REPORT)

    assert subject == "[ReviewPulse] Data drift detected in 2 feature(s): oov_rate, non_ascii_ratio"
    assert "oov_rate" in body and "0.9" in body
    assert "Requests in window:    120" in body
    assert "Features still in range:" in body and "chars" in body
    assert "GET /drift" in body


def test_drift_message_copes_with_a_sparse_report():
    subject, body = alerts.format_drift_message({"drifted_features": ["chars"]})

    assert "chars" in subject
    assert isinstance(body, str)


# ------------------------------------------------------------------ module hook


def test_alert_drift_sends_through_the_shared_manager():
    recorder = Recorder()
    alerts.set_manager(AlertManager(settings(), sender=recorder, threaded=False))

    assert alerts.alert_drift(DRIFT_REPORT) is True

    assert recorder.sent[0][0].startswith("[ReviewPulse] Data drift detected")


def test_alert_drift_never_raises(caplog):
    class Exploding:
        def notify(self, *args):
            raise RuntimeError("boom")

    alerts.set_manager(Exploding())

    with caplog.at_level(logging.ERROR, logger="reviewpulse.alerts"):
        assert alerts.alert_drift(DRIFT_REPORT) is False

    assert any("Drift alert failed" in r.getMessage() for r in caplog.records)


def test_shared_manager_reads_the_environment(monkeypatch):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ALERT_COOLDOWN_SECONDS", "120")

    manager = alerts.get_manager()

    assert manager.settings.username == "sender@example.com"
    assert manager.cooldown_seconds == 120
    assert alerts.get_manager() is manager
