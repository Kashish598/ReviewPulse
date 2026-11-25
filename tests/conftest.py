import os

import pytest

from src.logging_config import reset_logging


@pytest.fixture(scope="session", autouse=True)
def _keep_logs_out_of_the_repo(tmp_path_factory):
    """The API sets up file logging when it starts; send it to a temp folder during tests."""
    previous = os.environ.get("LOG_DIR")
    os.environ["LOG_DIR"] = str(tmp_path_factory.mktemp("logs"))
    yield
    reset_logging()
    if previous is None:
        os.environ.pop("LOG_DIR", None)
    else:
        os.environ["LOG_DIR"] = previous


@pytest.fixture(scope="session", autouse=True)
def _no_saving_or_labelling_unless_a_test_asks():
    """Tests must never write to a real database or call the Groq API.

    Saving is switched off and the key is removed for the whole run. Tests that need a store
    install their own temporary one (see tests/test_app_feedback.py).
    """
    saved = {name: os.environ.get(name) for name in ("FEEDBACK_STORE", "GROQ_API_KEY")}
    os.environ["FEEDBACK_STORE"] = "off"
    os.environ.pop("GROQ_API_KEY", None)
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
