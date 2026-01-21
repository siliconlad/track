"""Track - ML Experiment Tracking Library.

A library for logging machine learning experiments built on top of mcap.
Supports logging of text messages, images, point clouds, and more.

Example:
    >>> from track import Logger, LogLevel
    >>> with Logger("experiment.mcap") as logger:
    ...     logger.info("Starting experiment")
    ...     logger.log_image("input", image_bytes, format="png")
    ...     logger.log_pointcloud("lidar", points)
"""

from track.logger import Logger, LogLevel, NumericType, PointCloudField

__version__ = "0.1.0"
__all__ = [
    "Logger",
    "LogLevel",
    "NumericType",
    "PointCloudField",
]
