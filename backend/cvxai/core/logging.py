"""Logging setup and the per-request correlation id."""
from __future__ import annotations

import contextlib
import io
import logging
import sys
import uuid
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("cvxai_request_id", default="-")

_FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str:
    return _request_id.get()


class _StreamToLogger(io.TextIOBase):
    """File-like object that forwards whole lines to a logger."""

    def __init__(self, logger: logging.Logger, level: int, prefix: str = "") -> None:
        self._logger = logger
        self._level = level
        self._prefix = prefix
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self._logger.log(self._level, "%s%s", self._prefix, line)
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._logger.log(self._level, "%s%s", self._prefix, self._buffer.strip())
        self._buffer = ""


@contextlib.contextmanager
def capture_stdout(logger: logging.Logger, level: int = logging.INFO, prefix: str = ""):
    """Route `print()` from third-party code into the structured log.

    The components report their load progress with bare `print()`. Left alone,
    those lines land in the terminal unformatted and without the request id,
    which makes a startup log that is half structured and half not. Capturing
    them keeps one format for everything the service emits.
    """
    stream = _StreamToLogger(logger, level, prefix)
    with contextlib.redirect_stdout(stream):
        try:
            yield
        finally:
            stream.flush()


def configure_logging(level: str = "INFO") -> None:
    """Install a single stdout handler carrying the request id.

    Idempotent: repeated calls (reload, test-suite) replace the handler rather
    than stacking duplicates.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(_RequestIdFilter())
    root.addHandler(handler)

    for noisy in ("matplotlib", "PIL", "transformers", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Uvicorn's access logger would print a second, unformatted line for every
    # request that the request-context middleware already logs with its id and
    # timing. One line per request, not two.
    logging.getLogger("uvicorn.access").disabled = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
