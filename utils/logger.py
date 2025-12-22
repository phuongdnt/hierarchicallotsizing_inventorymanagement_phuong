"""
logger.py
=========

Simple logging utilities for the inventory management RL project.  This
module wraps Python's built-in :mod:`logging` module to provide
consistent formatting and optional file logging.  It can be extended
to support TensorBoard summaries or other tracking systems.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(name: str = "inventory_rl", log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Name of the logger.
        log_dir: Optional directory to write a log file.  If None,
            logging is only printed to the console.
        level: Logging level (e.g. logging.INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # File handler
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
