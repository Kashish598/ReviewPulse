import logging
import warnings

import pytest

from src import logging_config
from src.logging_config import LOG_FILE_NAME, parse_level, reset_logging, setup_logging


@pytest.fixture(autouse=True)
def clean_logging():
    reset_logging()
    yield
    reset_logging()


def read_log(path):
    for handler in logging.getLogger().handlers:
        handler.flush()
    return path.read_text(encoding="utf-8")


def test_parse_level_accepts_names_numbers_and_flags_bad_values():
    assert parse_level("debug") == (logging.DEBUG, True)
    assert parse_level("WARNING") == (logging.WARNING, True)
    assert parse_level("10") == (10, True)
    assert parse_level(None) == (logging.INFO, True)
    assert parse_level("loud") == (logging.INFO, False)


def test_creates_the_log_file_and_writes_records(tmp_path):
    log_file = setup_logging(log_dir=tmp_path, console=False, env={})

    logging.getLogger("src.demo").info("hello from the app")

    assert log_file == tmp_path / LOG_FILE_NAME
    assert "hello from the app" in read_log(log_file)


def test_record_format_has_time_level_logger_and_function(tmp_path):
    log_file = setup_logging(log_dir=tmp_path, console=False, env={})

    logging.getLogger("src.demo").warning("format check")

    line = [ln for ln in read_log(log_file).splitlines() if "format check" in ln][0]
    assert "WARNING" in line
    assert "src.demo" in line
    assert "test_record_format_has_time_level_logger_and_function" in line


def test_log_level_from_the_environment(tmp_path):
    log_file = setup_logging(log_dir=tmp_path, console=False, env={"LOG_LEVEL": "DEBUG"})
    logging.getLogger("src.demo").debug("debug detail")
    assert "debug detail" in read_log(log_file)

    reset_logging()
    log_file = setup_logging(log_dir=tmp_path, console=False, env={"LOG_LEVEL": "INFO"})
    logging.getLogger("src.demo").debug("hidden detail")
    assert "hidden detail" not in read_log(log_file)


def test_invalid_level_falls_back_to_info_and_says_so(tmp_path):
    log_file = setup_logging(log_dir=tmp_path, console=False, env={"LOG_LEVEL": "loud"})

    logging.getLogger("src.demo").debug("hidden")
    text = read_log(log_file)

    assert "Unknown LOG_LEVEL 'loud'" in text
    assert "hidden" not in text


def test_libraries_are_captured_and_debug_needs_lib_level(tmp_path):
    log_file = setup_logging(log_dir=tmp_path, console=False, env={"LOG_LEVEL": "DEBUG"})
    library = logging.getLogger("urllib3.connectionpool")
    library.info("library info line")
    library.debug("library debug line")
    text = read_log(log_file)
    assert "library info line" in text
    assert "library debug line" not in text

    reset_logging()
    log_file = setup_logging(log_dir=tmp_path, console=False, env={"LIB_LOG_LEVEL": "DEBUG"})
    logging.getLogger("urllib3.connectionpool").debug("now visible")
    assert "now visible" in read_log(log_file)


def test_a_library_logger_that_does_not_propagate_still_reaches_the_file(tmp_path):
    transformers = logging.getLogger("transformers")
    transformers.propagate = False  # what the real library does
    try:
        log_file = setup_logging(log_dir=tmp_path, console=False, env={})
        transformers.warning("weights loaded with warnings")

        assert "weights loaded with warnings" in read_log(log_file)
    finally:
        transformers.propagate = True


def test_calling_setup_twice_does_not_duplicate_lines(tmp_path):
    setup_logging(log_dir=tmp_path, console=False, env={})
    log_file = setup_logging(log_dir=tmp_path, console=False, env={})

    logging.getLogger("src.demo").info("only once please")

    assert read_log(log_file).count("only once please") == 1


def test_exception_traceback_is_logged_and_execution_continues(tmp_path):
    log_file = setup_logging(log_dir=tmp_path, console=False, env={})
    log = logging.getLogger("src.demo")

    try:
        {}["missing"]
    except KeyError:
        log.exception("lookup failed")
    log.info("still running")

    text = read_log(log_file)
    assert "lookup failed" in text
    assert "Traceback" in text
    assert "KeyError: 'missing'" in text
    assert "still running" in text


def test_unwritable_log_dir_falls_back_to_the_console(tmp_path):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("a file, so it cannot be used as a folder")

    log_file = setup_logging(log_dir=blocker, console=False, env={})

    assert log_file is None
    logging.getLogger("src.demo").info("the app keeps running")


def test_python_warnings_are_routed_into_the_log(tmp_path):
    log_file = setup_logging(log_dir=tmp_path, console=False, env={})

    warnings.warn("a library warning", UserWarning, stacklevel=1)

    assert "a library warning" in read_log(log_file)


def test_file_rotation_is_configured(tmp_path):
    setup_logging(log_dir=tmp_path, console=False, env={})

    handlers = [h for h in logging.getLogger().handlers if getattr(h, "_reviewpulse_handler", 0)]

    assert handlers[0].maxBytes == logging_config.MAX_LOG_BYTES
    assert handlers[0].backupCount == logging_config.LOG_BACKUPS


def test_reset_removes_our_handlers_and_restores_levels(tmp_path):
    before = logging.getLogger().level

    setup_logging(log_dir=tmp_path, console=False, env={"LIB_LOG_LEVEL": "DEBUG"})
    reset_logging()

    assert logging.getLogger().level == before
    assert not [h for h in logging.getLogger().handlers if getattr(h, "_reviewpulse_handler", 0)]
