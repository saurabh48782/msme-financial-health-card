from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

import src.utils.logger as logger_module
from src.utils.logger import get_logger, setup_logging


@pytest.fixture(autouse=True)
def _restore_logging() -> Any:
    """Every test reconfigures the root logger, so put it back afterwards."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    configured = logger_module._configured
    log_dir = logger_module.LOG_DIR
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)
    logger_module._configured = configured
    logger_module.LOG_DIR = log_dir


def _point_log_dir_at_an_unwritable_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ask for file logging under a directory whose parent is a regular file."""
    monkeypatch.setenv("LOG_TARGET", "file")
    monkeypatch.setattr(logger_module, "LOG_DIR", tmp_path / "file" / "logs")
    (tmp_path / "file").write_text("not a directory")


class TestFileHandlerFailure:
    def test_an_unwritable_log_dir_does_not_stop_start_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _point_log_dir_at_an_unwritable_path(tmp_path, monkeypatch)

        setup_logging(force=True)

        assert logging.getLogger().handlers, "stderr logging must survive"

    def test_the_failure_is_announced_rather_than_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Degrading to stderr-only is not something to discover during an outage."""
        _point_log_dir_at_an_unwritable_path(tmp_path, monkeypatch)
        monkeypatch.setenv("LOG_FORMAT", "json")

        setup_logging(force=True)

        assert "File logging disabled" in capsys.readouterr().err

    def test_stdout_target_skips_the_file_handler_entirely(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOG_TARGET", "stdout")
        monkeypatch.setattr(logger_module, "LOG_DIR", tmp_path / "logs")

        setup_logging(force=True)

        assert not (tmp_path / "logs").exists()
        assert not [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]


class TestIdempotence:
    def test_an_explicit_level_reconfigures_an_already_configured_logger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``get_logger`` configures on first import, long before an entry point
        asks for a level — ignoring that level would be a silent no-op."""
        monkeypatch.setenv("LOG_TARGET", "stdout")
        get_logger("warm-up")

        setup_logging("DEBUG")

        assert logging.getLogger().level == logging.DEBUG

    def test_a_bare_call_stays_a_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_TARGET", "stdout")
        setup_logging("DEBUG", force=True)

        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        setup_logging()

        assert logging.getLogger().level == logging.DEBUG
