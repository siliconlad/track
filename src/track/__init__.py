"""Track - ML Experiment Tracking Library.

A library for logging machine learning experiments built on top of pybag-sdk.
Supports logging of text messages, images, point clouds, and more.

Example (simple synchronous logging):
    >>> from track import Logger
    >>> with Logger("experiment.mcap") as logger:
    ...     logger.info("Starting experiment")
    ...     logger.log_image("input", image_array)  # numpy array or bytes

Example (non-blocking async logging for concurrent apps):
    >>> from track import AsyncLogger
    >>> with AsyncLogger("experiment.mcap") as logger:
    ...     logger.info("Non-blocking log call")
    ...     # Returns immediately, I/O happens in background

For multi-process applications, use AsyncLogger with use_process=True:
    >>> with AsyncLogger("experiment.mcap", use_process=True) as logger:
    ...     # Safe to use from multiple processes
"""

from track.concurrent import AsyncLogger
from track.logger import Logger, LogLevel

__version__ = "0.1.0"
__all__ = [
    "Logger",
    "LogLevel",
    "AsyncLogger",
]
