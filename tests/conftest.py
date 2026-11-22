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
