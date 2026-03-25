import logging

from src.utils.helpers import setup_logging


def test_setup_logging_skips_console_handler_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("LEAD_SCORING_DISABLE_CONSOLE_LOG", "1")

    logger = setup_logging(log_file=str(tmp_path / "train.log"))

    handler_types = {type(handler).__name__ for handler in logger.handlers}
    assert "FileHandler" in handler_types
    assert "StreamHandler" not in handler_types

    logger.handlers.clear()


def test_setup_logging_keeps_console_handler_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("LEAD_SCORING_DISABLE_CONSOLE_LOG", raising=False)

    logger = setup_logging(log_file=str(tmp_path / "train.log"))

    handler_types = {type(handler).__name__ for handler in logger.handlers}
    assert "FileHandler" in handler_types
    assert "StreamHandler" in handler_types

    logger.handlers.clear()
