"""
utils/logger.py
---------------
Unified logging utility. Writes to both console and a rotating log file.
Usage:
    from utils.logger import get_logger
    logger = get_logger("train", log_dir="outputs/logs")
    logger.info("Training started")
"""

import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "outputs/logs", level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a named logger that writes to console and file.

    Args:
        name:    Logger name (used as filename prefix).
        log_dir: Directory to write log files. Created if it does not exist.
        level:   Logging level. Default INFO.

    Returns:
        Configured Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid adding duplicate handlers on repeated calls
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler — one file per logger name + timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
