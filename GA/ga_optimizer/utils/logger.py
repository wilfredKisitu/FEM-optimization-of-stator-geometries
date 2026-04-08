"""utils/logger.py — Configure the root logger for the GA optimizer."""

from __future__ import annotations

import logging
import sys


def setup_logger(level: str = "INFO") -> logging.Logger:
    """Configure the root logger with a human-readable format.

    Parameters
    ----------
    level:
        Logging level string — ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, or
        ``"ERROR"``.

    Returns
    -------
    logging.Logger
        The ``ga_optimizer`` package logger.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)

    log = logging.getLogger("ga_optimizer")
    if log.handlers:
        # Already configured — just update the level
        log.setLevel(numeric)
        return log

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric)
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)

    log.setLevel(numeric)
    log.addHandler(handler)
    log.propagate = False   # don't double-print to root logger

    return log
