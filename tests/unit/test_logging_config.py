import logging

from shared.logging_config import configure_logging


def test_configure_logging_does_not_crash():
    configure_logging("INFO")


def test_log_level_info():
    configure_logging("INFO")
    assert logging.getLogger().level == logging.INFO


def test_log_level_debug():
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_log_level_warning():
    configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_invalid_level_falls_back_to_info():
    configure_logging("NOTAREAL_LEVEL")
    assert logging.getLogger().level == logging.INFO


def test_repeated_calls_do_not_duplicate_handlers():
    configure_logging("INFO")
    configure_logging("INFO")
    configure_logging("INFO")
    # configure_logging clears handlers before adding one, so only one should exist
    assert len(logging.getLogger().handlers) == 1
