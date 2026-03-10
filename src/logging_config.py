"""
Centralised logging configuration for the MLB HR model pipeline.

Usage in entry-point scripts (build_features_season.py, train.py, etc.):

    from src.logging_config import configure_logging
    configure_logging()          # INFO to console
    configure_logging(level=logging.DEBUG)   # verbose
    configure_logging(log_file="build.log")  # also write to file

Usage in library modules (never call configure_logging here — just get a logger):

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Loading events %s -> %s", start, end)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    fmt: str = "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt: str = "%H:%M:%S",
) -> None:
    """
    Configure root logger for the pipeline.

    Parameters
    ----------
    level    : logging level for console handler (default INFO).
    log_file : optional path; if given, also writes to that file at DEBUG level.
    fmt      : log format string.
    datefmt  : date/time format string.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # root captures everything; handlers filter

    # Avoid duplicate handlers if called more than once (e.g., in tests)
    if root.handlers:
        root.handlers.clear()

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Optional file handler (always DEBUG so the file is verbose)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "pybaseball", "lightgbm", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
