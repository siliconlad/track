"""Track - ML Experiment Tracking Library."""

from __future__ import annotations

import threading
from pathlib import Path
import numpy as np

from track.logger import Logger, LogLevel

__version__ = "0.2.0"
__all__ = [
    "Logger",
    "LogLevel",
    # Registry
    "get_logger",
    # Default logger lifecycle
    "init",
    "finish",
    # Convenience log functions
    "debug",
    "info",
    "warning",
    "error",
    "fatal",
    # Convenience data functions
    "log_image",
    "log_pointcloud",
    "add_metadata",
    "add_attachment",
]

# ---------------------------------------------------------------------------
# Global logger registry
# ---------------------------------------------------------------------------

_loggers: dict[str, Logger] = {}
_default_logger: Logger | None = None
_registry_lock = threading.Lock()


def get_logger(name: str) -> Logger:
    """Retrieve a registered logger by name.

    Raises:
        KeyError: If no logger with that name has been registered via ``init``.
    """
    with _registry_lock:
        try:
            return _loggers[name]
        except KeyError:
            raise KeyError(
                f"No logger named {name!r}. Create one with track.init({name!r}, ...)."
            ) from None


def init(
    name: str,
    output_dir: str | Path | None = None,
    *,
    use_process: bool = False,
) -> Logger:
    """Create, open, register, and set a logger as the default.

    Args:
        name: Logger name (included in log messages).
        output_dir: Directory where MCAP files should be written.
        use_process: If True, use a background writer process.

    Returns:
        The newly created and opened ``Logger``.

    Raises:
        ValueError: If a logger with *name* is already registered.
    """
    global _default_logger

    with _registry_lock:
        if name in _loggers:
            raise ValueError(
                f"A logger named {name!r} is already registered. "
                "Call track.finish() first or use a different name."
            )

    logger = Logger(name, output_dir, use_process=use_process)
    logger.open()

    with _registry_lock:
        # Re-check after releasing the lock for open() — another thread
        # could have registered the same name in the meantime.
        if name in _loggers:
            logger.close()
            raise ValueError(
                f"A logger named {name!r} was registered concurrently. "
                "Call track.finish() first or use a different name."
            )
        _loggers[name] = logger
        _default_logger = logger

    return logger


def finish() -> None:
    """Close and unregister the default logger."""
    global _default_logger

    with _registry_lock:
        logger = _default_logger
        if logger is None:
            return
        _default_logger = None
        _loggers.pop(logger._name, None)

    logger.close()


# ---------------------------------------------------------------------------
# Convenience functions — delegate to the default logger
# ---------------------------------------------------------------------------

def _get_default() -> Logger:
    """Return the default logger or raise a clear error."""
    logger = _default_logger
    if logger is None:
        raise RuntimeError(
            "No default logger. Call track.init(...) first."
        )
    return logger


def debug(message: str, *, timestamp_ns: int | None = None) -> None:
    """Log a debug message to the default logger."""
    _get_default().debug(message, timestamp_ns=timestamp_ns, _stacklevel=3)


def info(message: str, *, timestamp_ns: int | None = None) -> None:
    """Log an info message to the default logger."""
    _get_default().info(message, timestamp_ns=timestamp_ns, _stacklevel=3)


def warning(message: str, *, timestamp_ns: int | None = None) -> None:
    """Log a warning message to the default logger."""
    _get_default().warning(message, timestamp_ns=timestamp_ns, _stacklevel=3)


def error(message: str, *, timestamp_ns: int | None = None) -> None:
    """Log an error message to the default logger."""
    _get_default().error(message, timestamp_ns=timestamp_ns, _stacklevel=3)


def fatal(message: str, *, timestamp_ns: int | None = None) -> None:
    """Log a fatal message to the default logger."""
    _get_default().fatal(message, timestamp_ns=timestamp_ns, _stacklevel=3)


def log_image(
    topic: str,
    image: bytes | np.ndarray,
    *,
    format: str = "png",
    frame_id: str | None = None,
    timestamp_ns: int | None = None,
) -> None:
    """Log an image to the default logger."""
    _get_default().log_image(
        topic, image, format=format, frame_id=frame_id, timestamp_ns=timestamp_ns,
    )


def log_pointcloud(
    topic: str,
    points: np.ndarray,
    *,
    frame_id: str = "world",
    timestamp_ns: int | None = None,
) -> None:
    """Log a point cloud to the default logger."""
    _get_default().log_pointcloud(
        topic, points, frame_id=frame_id, timestamp_ns=timestamp_ns,
    )


def add_metadata(name: str, data: dict[str, str]) -> None:
    """Add metadata to the default logger."""
    _get_default().add_metadata(name, data)


def add_attachment(
    name: str,
    data: bytes,
    media_type: str = "application/octet-stream",
    *,
    timestamp_ns: int | None = None,
) -> None:
    """Add an attachment to the default logger."""
    _get_default().add_attachment(
        name, data, media_type=media_type, timestamp_ns=timestamp_ns,
    )
