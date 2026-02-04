"""Logging configuration for the autonomous navigation system.

This module sets up structured logging with:
- Console output with colored formatting (INFO+)
- File output with detailed formatting (DEBUG+)
- Configurable log levels and output paths
"""

import logging
import os
from pathlib import Path
from typing import Optional

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


def setup_logger(
    name: str = "navigation",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_to_console: bool = True
) -> logging.Logger:
    """Set up logger with console and file handlers.

    Args:
        name: Logger name (module name)
        level: Log level - DEBUG, INFO, WARNING, ERROR
        log_file: Path to log file (None = no file logging)
        log_to_console: Whether to output to console

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("navigation", "INFO", "logs/app.log")
        >>> logger.info("Processing started")
        >>> logger.warning("Frame 45 detection failed")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler with color (INFO+)
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        if HAS_COLORLOG:
            # Colored formatter for console
            console_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(levelname)-8s%(reset)s %(cyan)s%(name)s%(reset)s: %(message)s",
                log_colors={
                    'DEBUG': 'blue',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            # Fallback to plain formatter
            console_formatter = logging.Formatter(
                "%(levelname)-8s %(name)s: %(message)s"
            )

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler with detailed output (DEBUG+)
    if log_file:
        # Create log directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Detailed formatter for file
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger by name.

    Args:
        name: Logger name (usually module name)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)
