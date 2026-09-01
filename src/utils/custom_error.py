"""A single exception type for pipeline failures, carrying the failing location."""

from __future__ import annotations

import sys
from types import TracebackType


def _format(error: BaseException, exc_tb: TracebackType | None) -> str:
    """Render ``error`` with the file and line of the *deepest* frame that raised."""
    if exc_tb is None:
        return f"{type(error).__name__}: {error}"
    tb = exc_tb
    while tb.tb_next is not None:
        tb = tb.tb_next
    filename = tb.tb_frame.f_code.co_filename
    return f"Error in [{filename}] at line [{tb.tb_lineno}]: {type(error).__name__}: {error}"


class CustomError(Exception):
    """Wraps an underlying exception with the file/line where it originated."""

    def __init__(self, error: BaseException | str) -> None:
        if isinstance(error, str):
            self.original: BaseException | None = None
            message = error
        else:
            self.original = error
            _, _, exc_tb = sys.exc_info()
            message = _format(error, exc_tb if exc_tb is not None else error.__traceback__)
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message
