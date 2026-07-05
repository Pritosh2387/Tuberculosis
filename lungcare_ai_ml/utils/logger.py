"""
Logging configuration and factory for LungCare AI.

Provides named loggers under the 'lungcare.*' namespace with
rotating file handlers, console output, and test-safe reset.
All internal package modules obtain their loggers via the stdlib
``logging.getLogger("lungcare.<module>")`` pattern so they
inherit from the root 'lungcare' logger configured by
:func:`setup_logging`.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
)
DEFAULT_DATEFMT: str = "%Y-%m-%d %H:%M:%S"

_ROOT_CONFIGURED: bool = False
_NAMED_LOGGERS: dict[str, logging.Logger] = {}


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = False,
    log_file: str | Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
) -> logging.Logger:
    """
    Configure the root 'lungcare' logger for the entire application.

    Must be called once at application startup (training script, inference
    entry point, etc.).  All child loggers (``lungcare.checkpoint``,
    ``lungcare.dicom``, etc.) inherit handlers through propagation.

    Subsequent calls within the same process are no-ops; use
    :func:`reset_logging` in tests to force re-configuration.

    Args:
        level: Logging level string (``'DEBUG'``, ``'INFO'``, ``'WARNING'``,
            ``'ERROR'``, ``'CRITICAL'``).
        log_to_file: Whether to attach a :class:`RotatingFileHandler`.
        log_file: Path to the log file.  Required when *log_to_file* is True.
        max_bytes: Maximum bytes per log file before rotation.
        backup_count: Number of rotated backup files to keep.
        fmt: ``logging.Formatter`` format string.
        datefmt: Date/time format for the formatter.

    Returns:
        The configured root ``'lungcare'`` :class:`logging.Logger`.
    """
    global _ROOT_CONFIGURED

    root = logging.getLogger("lungcare")
    if _ROOT_CONFIGURED:
        return root

    root.handlers.clear()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)
    root.propagate = False

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_to_file and log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    _ROOT_CONFIGURED = True
    return root


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: str | Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
) -> logging.Logger:
    """
    Get or create a fully self-contained named logger.

    Unlike child loggers that rely on root propagation, loggers created
    here carry their own handlers.  Intended for scripts, CLI entry points,
    and external code that does **not** call :func:`setup_logging`.

    Repeated calls with the same *name* return the cached logger without
    adding duplicate handlers.

    Args:
        name: Logger name. Automatically prefixed with ``'lungcare.'`` if
            the prefix is absent.
        level: Logging level string.
        log_file: Optional path to a rotating log file.
        max_bytes: Maximum bytes per log file before rotation.
        backup_count: Number of backup log files.
        fmt: Log record format string.
        datefmt: Date/time format string.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    qualified = name if name.startswith("lungcare") else f"lungcare.{name}"
    if qualified in _NAMED_LOGGERS:
        return _NAMED_LOGGERS[qualified]

    logger = logging.getLogger(qualified)
    logger.handlers.clear()

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    _NAMED_LOGGERS[qualified] = logger
    return logger


def reset_logging() -> None:
    """
    Reset all LungCare AI loggers to a pristine state.

    Clears all handlers from the ``'lungcare'`` root and every
    ``'lungcare.*'`` child logger, and purges the internal caches.

    Intended for **test isolation only**.  Do not call in production code.
    """
    global _ROOT_CONFIGURED

    _NAMED_LOGGERS.clear()
    _ROOT_CONFIGURED = False

    root = logging.getLogger("lungcare")
    root.handlers.clear()

    manager = logging.Logger.manager
    for logger_name, logger_obj in list(manager.loggerDict.items()):
        if logger_name.startswith("lungcare") and isinstance(
            logger_obj, logging.Logger
        ):
            logger_obj.handlers.clear()
