"""ML Experiment Logger built on pybag-sdk."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import IO, Any

import numpy as np
from pybag.mcap_writer import McapFileWriter
from pybag.ros2.humble import builtin_interfaces, sensor_msgs, std_msgs
import pybag.types as t


class LogLevel(IntEnum):
    """Log severity levels."""

    UNKNOWN = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    FATAL = 5


# Custom Log message type (similar to rcl_interfaces/Log)
@dataclass(kw_only=True)
class Log:
    """Log message for ML experiment tracking."""

    __msg_name__ = "track/msg/Log"

    stamp: builtin_interfaces.Time
    level: t.uint8
    name: t.string
    msg: t.string
    file: t.string
    line: t.uint32


# Dtype to PointField datatype mapping
DTYPE_TO_POINTFIELD: dict[np.dtype, int] = {
    np.dtype("int8"): sensor_msgs.PointField.INT8,
    np.dtype("uint8"): sensor_msgs.PointField.UINT8,
    np.dtype("int16"): sensor_msgs.PointField.INT16,
    np.dtype("uint16"): sensor_msgs.PointField.UINT16,
    np.dtype("int32"): sensor_msgs.PointField.INT32,
    np.dtype("uint32"): sensor_msgs.PointField.UINT32,
    np.dtype("float32"): sensor_msgs.PointField.FLOAT32,
    np.dtype("float64"): sensor_msgs.PointField.FLOAT64,
}


def _time_ns() -> int:
    """Get current time in nanoseconds."""
    return time.time_ns()


def _ns_to_stamp(ns: int) -> builtin_interfaces.Time:
    """Convert nanoseconds to builtin_interfaces.Time."""
    return builtin_interfaces.Time(
        sec=ns // 1_000_000_000,
        nanosec=ns % 1_000_000_000,
    )


def _make_header(timestamp_ns: int, frame_id: str = "") -> std_msgs.Header:
    """Create a std_msgs.Header."""
    return std_msgs.Header(
        stamp=_ns_to_stamp(timestamp_ns),
        frame_id=frame_id,
    )


class Logger:
    """ML Experiment Logger for tracking logs, images, and point clouds.

    This logger writes data to MCAP files using ROS2 message types,
    allowing visualization in Foxglove Studio and other compatible tools.

    Example:
        >>> with Logger("experiment.mcap") as logger:
        ...     logger.info("Starting training")
        ...     logger.log_image("input", image_bytes, format="png")
        ...     logger.warning("Learning rate might be too high")
    """

    def __init__(
        self,
        output: str | Path,
        *,
        name: str = "track",
        chunk_size: int = 1024 * 1024,
        compression: str = "lz4",
    ) -> None:
        """Initialize the logger.

        Args:
            output: Output file path.
            name: Logger name (included in log messages).
            chunk_size: Size of data chunks in bytes.
            compression: Compression type ('zstd', 'lz4', or 'none').
        """
        self._name = name
        self._output = Path(output)
        self._chunk_size = chunk_size
        self._compression = compression

        self._writer: McapFileWriter | None = None

    def __enter__(self) -> Logger:
        """Enter context manager."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.close()

    def open(self) -> None:
        """Open the logger for writing."""
        if self._writer is not None:
            return

        self._writer = McapFileWriter.open(
            self._output,
            mode="w",
            profile="ros2",
            chunk_size=self._chunk_size,
            chunk_compression=self._compression,  # type: ignore
        )

    def close(self) -> None:
        """Close the logger and finalize the file."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _ensure_open(self) -> None:
        """Ensure the logger is open."""
        if self._writer is None:
            raise RuntimeError("Logger is not open. Call open() or use as context manager.")

    def _get_caller_info(self) -> tuple[str, int]:
        """Get the file and line number of the caller."""
        import inspect

        frame = inspect.currentframe()
        try:
            # Skip: _get_caller_info -> _log -> debug/info/etc -> actual caller
            for _ in range(4):
                if frame is not None:
                    frame = frame.f_back
            if frame is not None:
                return frame.f_code.co_filename, frame.f_lineno
        finally:
            del frame
        return "", 0

    def _log(
        self,
        level: LogLevel,
        message: str,
        *,
        timestamp_ns: int | None = None,
        file: str = "",
        line: int = 0,
    ) -> None:
        """Internal log method."""
        self._ensure_open()
        assert self._writer is not None

        ts = timestamp_ns if timestamp_ns is not None else _time_ns()

        log_msg = Log(
            stamp=_ns_to_stamp(ts),
            level=int(level),
            name=self._name,
            msg=message,
            file=file,
            line=line,
        )

        self._writer.write_message("/log", ts, log_msg)

    def debug(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log a debug message.

        Args:
            message: The message to log.
            timestamp_ns: Optional timestamp in nanoseconds. Defaults to current time.
        """
        file, line = self._get_caller_info()
        self._log(LogLevel.DEBUG, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def info(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log an info message.

        Args:
            message: The message to log.
            timestamp_ns: Optional timestamp in nanoseconds. Defaults to current time.
        """
        file, line = self._get_caller_info()
        self._log(LogLevel.INFO, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def warning(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log a warning message.

        Args:
            message: The message to log.
            timestamp_ns: Optional timestamp in nanoseconds. Defaults to current time.
        """
        file, line = self._get_caller_info()
        self._log(LogLevel.WARNING, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def error(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log an error message.

        Args:
            message: The message to log.
            timestamp_ns: Optional timestamp in nanoseconds. Defaults to current time.
        """
        file, line = self._get_caller_info()
        self._log(LogLevel.ERROR, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def fatal(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log a fatal message.

        Args:
            message: The message to log.
            timestamp_ns: Optional timestamp in nanoseconds. Defaults to current time.
        """
        file, line = self._get_caller_info()
        self._log(LogLevel.FATAL, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def log_image(
        self,
        topic: str,
        image: bytes | np.ndarray,
        *,
        format: str = "png",
        frame_id: str = "",
        timestamp_ns: int | None = None,
    ) -> None:
        """Log an image.

        Args:
            topic: Topic name for the image (e.g., "camera/rgb").
            image: Image data - either compressed bytes (PNG, JPEG, WebP) or
                   a numpy array (HxW for grayscale, HxWx3 for RGB, HxWx4 for RGBA).
            format: Image format ('png', 'jpeg', or 'webp'). Used when encoding arrays.
            frame_id: Frame of reference for the image.
            timestamp_ns: Optional timestamp in nanoseconds. Defaults to current time.
        """
        self._ensure_open()
        assert self._writer is not None

        ts = timestamp_ns if timestamp_ns is not None else _time_ns()
        channel_topic = f"/images/{topic}" if not topic.startswith("/") else topic

        # Convert numpy array to bytes if needed
        if isinstance(image, np.ndarray):
            image = self._encode_image_array(image, format)

        msg = sensor_msgs.CompressedImage(
            header=_make_header(ts, frame_id or topic),
            format=format,
            data=list(image),
        )

        self._writer.write_message(channel_topic, ts, msg)

    def _encode_image_array(self, array: np.ndarray, format: str) -> bytes:
        """Encode a numpy array to compressed image bytes."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "PIL/Pillow is required for numpy array images. "
                "Install with: pip install Pillow"
            )

        # Determine PIL mode from array shape
        if array.ndim == 2:
            mode = "L"
        elif array.ndim == 3 and array.shape[2] == 3:
            mode = "RGB"
        elif array.ndim == 3 and array.shape[2] == 4:
            mode = "RGBA"
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")

        # Ensure uint8
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)

        img = Image.fromarray(array, mode=mode)

        # Encode to bytes
        buffer = io.BytesIO()
        img.save(buffer, format=format.upper())
        return buffer.getvalue()

    def log_pointcloud(
        self,
        topic: str,
        points: np.ndarray,
        *,
        frame_id: str = "world",
        timestamp_ns: int | None = None,
    ) -> None:
        """Log a point cloud.

        Args:
            topic: Topic name for the point cloud.
            points: Point cloud data as a numpy structured array (recarray) with N points.
                   Each field in the dtype becomes a PointField. Common field names:
                   - x, y, z: position coordinates
                   - r/red, g/green, b/blue, a/alpha: color components
                   - intensity: intensity values
                   - normal_x, normal_y, normal_z: surface normals

                   Example dtypes:
                   - XYZ only: np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
                   - With color: np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                           ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
            frame_id: Frame of reference.
            timestamp_ns: Optional timestamp in nanoseconds. Defaults to current time.

        Example:
            >>> # Create point cloud with XYZ and RGB
            >>> dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ...                   ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
            >>> points = np.zeros(100, dtype=dtype)
            >>> points['x'] = np.random.randn(100)
            >>> points['y'] = np.random.randn(100)
            >>> points['z'] = np.random.randn(100)
            >>> points['r'] = np.random.randint(0, 255, 100)
            >>> points['g'] = np.random.randint(0, 255, 100)
            >>> points['b'] = np.random.randint(0, 255, 100)
            >>> logger.log_pointcloud("lidar", points)
        """
        self._ensure_open()
        assert self._writer is not None

        ts = timestamp_ns if timestamp_ns is not None else _time_ns()
        channel_topic = f"/pointclouds/{topic}" if not topic.startswith("/") else topic

        # Validate input
        if not isinstance(points.dtype, np.dtype) or points.dtype.names is None:
            raise ValueError(
                "points must be a numpy structured array (recarray). "
                "Example: np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])"
            )

        # Build PointField descriptors from dtype
        fields = []
        for name in points.dtype.names:
            field_dtype = points.dtype.fields[name][0]
            offset = points.dtype.fields[name][1]

            # Map numpy dtype to PointField datatype
            base_dtype = np.dtype(field_dtype.base)
            if base_dtype not in DTYPE_TO_POINTFIELD:
                raise ValueError(f"Unsupported dtype for field '{name}': {base_dtype}")

            fields.append(
                sensor_msgs.PointField(
                    name=name,
                    offset=offset,
                    datatype=DTYPE_TO_POINTFIELD[base_dtype],
                    count=1,
                )
            )

        # Get point data as bytes
        n_points = len(points)
        point_step = points.dtype.itemsize
        row_step = n_points * point_step
        data = points.tobytes()

        msg = sensor_msgs.PointCloud2(
            header=_make_header(ts, frame_id),
            height=1,
            width=n_points,
            fields=fields,
            is_bigendian=False,
            point_step=point_step,
            row_step=row_step,
            data=list(data),
            is_dense=True,
        )

        self._writer.write_message(channel_topic, ts, msg)

    def add_metadata(self, name: str, data: dict[str, str]) -> None:
        """Add metadata to the MCAP file.

        Args:
            name: Metadata name/category.
            data: Key-value metadata pairs.
        """
        self._ensure_open()
        assert self._writer is not None
        self._writer.write_metadata(name, data)

    def add_attachment(
        self,
        name: str,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        timestamp_ns: int | None = None,
    ) -> None:
        """Add an attachment to the MCAP file.

        Attachments are useful for storing auxiliary data like model weights,
        configuration files, or other binary data.

        Args:
            name: Attachment filename.
            data: Attachment data.
            media_type: MIME type of the attachment.
            timestamp_ns: Optional timestamp in nanoseconds. Defaults to current time.
        """
        self._ensure_open()
        assert self._writer is not None

        ts = timestamp_ns if timestamp_ns is not None else _time_ns()
        self._writer.write_attachment(name, data, media_type=media_type, log_time=ts)
