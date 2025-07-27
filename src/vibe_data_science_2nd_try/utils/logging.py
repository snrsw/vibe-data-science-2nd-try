import logging
import sys
from pathlib import Path
from typing import Any, Optional

import structlog
from structlog.processors import TimeStamper
from structlog.stdlib import LoggerFactory


def configure_logging(
    level: str = "INFO",
    output_path: Optional[Path] = None,
    include_timestamp: bool = True,
) -> None:
    """
    Configure structlog for the application.

    Args:
        level: The log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        output_path: Path to output log file (logs to stdout if None)
        include_timestamp: Whether to include timestamps in log entries
    """
    log_level = getattr(logging, level)

    # Configure standard library logging
    handlers: list[Any] = [logging.StreamHandler(sys.stdout)]
    if output_path:
        handlers.append(logging.FileHandler(output_path))

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
    )

    # Configure processors for structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.dict_tracebacks,
        structlog.processors.StackInfoRenderer(),
    ]

    # Add timestamp if requested
    if include_timestamp:
        processors.append(TimeStamper(fmt="%Y-%m-%d %H:%M:%S"))

    # Add formatters
    processors.extend(
        [
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    )

    # Configure structlog
    structlog.configure(
        processors=processors,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "") -> structlog.stdlib.BoundLogger:
    """
    Get a configured structlog logger.

    Args:
        name: Optional name for the logger

    Returns:
        A configured structlog logger
    """
    return structlog.get_logger(name)
