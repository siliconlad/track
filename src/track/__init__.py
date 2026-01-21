"""Track - ML Experiment Tracking Library.

A library for logging machine learning experiments built on top of mcap.
Supports logging of text messages, images, point clouds, and more.

Example:
    >>> from track import Logger, LogLevel
    >>> with Logger("experiment.mcap") as logger:
    ...     logger.info("Starting experiment")
    ...     logger.log_image("input", image_bytes, format="png")
    ...     logger.log_pointcloud("lidar", points)

For concurrent logging, see:
    - ThreadSafeLogger: Thread-safe logging with locks
    - MultiProcessLogger: Process-safe logging with queues
    - ProcessLocalLogger: Per-process files for distributed training
    - merge_mcap_files: Merge multiple MCAP files
"""

from track.concurrent import (
    MultiProcessLogger,
    ProcessLocalLogger,
    ThreadSafeLogger,
    merge_mcap_files,
)
from track.logger import Logger, LogLevel, NumericType, PointCloudField

__version__ = "0.1.0"
__all__ = [
    # Basic logging
    "Logger",
    "LogLevel",
    "NumericType",
    "PointCloudField",
    # Concurrent logging
    "ThreadSafeLogger",
    "MultiProcessLogger",
    "ProcessLocalLogger",
    "merge_mcap_files",
]
