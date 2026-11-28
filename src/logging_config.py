# src/logging_config.py (K1).
# setup_logging() runs once at startup and writes every record to the console
# and to logs/reviewpulse.log (5 files of 5 MB).
# LOG_LEVEL sets the level for our own loggers, and LIB_LOG_LEVEL sets it for
# libraries (uvicorn, transformers, httpx, urllib3, mlflow).
# At DEBUG it logs what each stage received. A file that can't be created never stops the app.
# Tests: tests/test_logging_config.py.

"""Logging for the API and the scripts: the `logging` module, a rotating log file, no print().

Call `setup_logging()` once at start-up (the API does it in its lifespan). It

  * writes every record to logs/reviewpulse.log (rotating, 5 files of 5 MB) and to the console,
  * sets the level of our own loggers (`src.*`, `reviewpulse.*`) from LOG_LEVEL,
  * captures the libraries too (uvicorn, transformers, huggingface_hub, httpx, urllib3, mlflow),
    including the ones that do not propagate to the root logger, at LIB_LOG_LEVEL,
  * routes Python warnings into the log.

At LOG_LEVEL=DEBUG our modules also log what each stage received and returned.

Environment: LOG_LEVEL (default INFO), LIB_LOG_LEVEL (default INFO), LOG_DIR (default ./logs).
A log file that cannot be created never stops the app: it falls back to console only.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "logs"
LOG_FILE_NAME = "reviewpulse.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
MAX_LOG_BYTES = 5_000_000
LOG_BACKUPS = 5

APP_LOGGERS = ("src", "reviewpulse")
EXTERNAL_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "transformers",
    "huggingface_hub",
    "httpx",
    "httpcore",
    "urllib3",
    "mlflow",
)

_FLAG = "_reviewpulse_handler"
_saved_levels: dict[str, int] = {}
logger = logging.getLogger(__name__)


def parse_level(value: str | None, default: int = logging.INFO) -> tuple[int, bool]:
    """Turn 'debug', 'INFO' or '10' into a level. The flag is False when the value was invalid."""
    if value is None or not str(value).strip():
        return default, True
    text = str(value).strip()
    if text.isdigit():
        return int(text), True
    level = logging.getLevelName(text.upper())
    if isinstance(level, int):
        return level, True
    return default, False


def _all_loggers() -> list[logging.Logger]:
    names = list(APP_LOGGERS) + list(EXTERNAL_LOGGERS)
    return [logging.getLogger()] + [logging.getLogger(name) for name in names]


def _remove_our_handlers() -> None:
    for target in _all_loggers():
        for handler in list(target.handlers):
            if getattr(handler, _FLAG, False):
                target.removeHandler(handler)
                handler.close()


def reset_logging() -> None:
    """Undo setup_logging(): remove our handlers and restore the logger levels."""
    _remove_our_handlers()
    for name, level in _saved_levels.items():
        logging.getLogger(name or None).setLevel(level)
    _saved_levels.clear()
    logging.captureWarnings(False)


def _remember_and_set(target: logging.Logger, level: int) -> None:
    _saved_levels.setdefault(target.name if target.name != "root" else "", target.level)
    target.setLevel(level)


def _mark(handler: logging.Handler, formatter: logging.Formatter) -> logging.Handler:
    handler.setFormatter(formatter)
    setattr(handler, _FLAG, True)
    return handler


def setup_logging(
    level: str | int | None = None,
    library_level: str | int | None = None,
    log_dir: str | Path | None = None,
    console: bool = True,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Configure logging and return the log file path (None when only the console is used)."""
    env = os.environ if env is None else env
    problems: list[str] = []

    def resolve(argument, variable: str) -> int:
        raw = argument if argument is not None else env.get(variable)
        if isinstance(raw, int):
            return raw
        value, valid = parse_level(raw)
        if not valid:
            problems.append(f"Unknown {variable} {raw!r}, using INFO")
        return value

    app_level = resolve(level, "LOG_LEVEL")
    lib_level = resolve(library_level, "LIB_LOG_LEVEL")

    _remove_our_handlers()
    formatter = logging.Formatter(LOG_FORMAT)

    log_path: Path | None = None
    file_handler: logging.Handler | None = None
    directory = Path(log_dir or env.get("LOG_DIR") or DEFAULT_LOG_DIR)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / LOG_FILE_NAME
        file_handler = _mark(
            logging.handlers.RotatingFileHandler(
                log_path, maxBytes=MAX_LOG_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8"
            ),
            formatter,
        )
    except OSError as error:
        log_path = None
        problems.append(f"Cannot write logs to {directory} ({error}), logging to the console only")

    root = logging.getLogger()
    _remember_and_set(root, lib_level)
    if file_handler is not None:
        root.addHandler(file_handler)
    if console:
        root.addHandler(_mark(logging.StreamHandler(), formatter))

    for name in APP_LOGGERS:
        _remember_and_set(logging.getLogger(name), app_level)

    # Libraries that keep their own handlers and do not propagate (uvicorn, transformers,
    # mlflow) would never reach the root logger, so give them the file handler directly.
    for name in EXTERNAL_LOGGERS:
        external = logging.getLogger(name)
        _remember_and_set(external, lib_level)
        if file_handler is not None and not external.propagate:
            external.addHandler(file_handler)

    logging.captureWarnings(True)

    for problem in problems:
        logger.warning(problem)
    logger.info(
        "Logging ready: app level %s, library level %s, file %s",
        logging.getLevelName(app_level),
        logging.getLevelName(lib_level),
        log_path,
    )
    return log_path
