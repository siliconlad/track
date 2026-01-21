"""ML Experiment Logger built on mcap."""

from __future__ import annotations

import base64
import inspect
import io
import json
import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import IO, Any, Sequence

import numpy as np
from mcap.writer import Writer

from track.schemas import (
    COMPRESSED_IMAGE_SCHEMA,
    LOG_SCHEMA,
    POINT_CLOUD_SCHEMA,
    get_schema_json,
)


class LogLevel(IntEnum):
    """Log severity levels compatible with Foxglove."""

    UNKNOWN = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    FATAL = 5


class NumericType(IntEnum):
    """Numeric types for point cloud fields."""

    UNKNOWN = 0
    UINT8 = 1
    INT8 = 2
    UINT16 = 3
    INT16 = 4
    UINT32 = 5
    INT32 = 6
    FLOAT32 = 7
    FLOAT64 = 8


# Mapping from NumericType to struct format and size
NUMERIC_TYPE_INFO: dict[NumericType, tuple[str, int]] = {
    NumericType.UINT8: ("B", 1),
    NumericType.INT8: ("b", 1),
    NumericType.UINT16: ("H", 2),
    NumericType.INT16: ("h", 2),
    NumericType.UINT32: ("I", 4),
    NumericType.INT32: ("i", 4),
    NumericType.FLOAT32: ("f", 4),
    NumericType.FLOAT64: ("d", 8),
}

# Mapping from numpy dtype to NumericType
NUMPY_TO_NUMERIC_TYPE: dict[np.dtype, NumericType] = {
    np.dtype("uint8"): NumericType.UINT8,
    np.dtype("int8"): NumericType.INT8,
    np.dtype("uint16"): NumericType.UINT16,
    np.dtype("int16"): NumericType.INT16,
    np.dtype("uint32"): NumericType.UINT32,
    np.dtype("int32"): NumericType.INT32,
    np.dtype("float32"): NumericType.FLOAT32,
    np.dtype("float64"): NumericType.FLOAT64,
}


@dataclass
class PointCloudField:
    """Definition of a field in a point cloud."""

    name: str
    offset: int
    numeric_type: NumericType


def _time_ns() -> int:
    """Get current time in nanoseconds."""
    return time.time_ns()


def _ns_to_timestamp(ns: int) -> dict[str, int]:
    """Convert nanoseconds to Foxglove timestamp format."""
    return {"sec": ns // 1_000_000_000, "nsec": ns % 1_000_000_000}


class Logger:
    """ML Experiment Logger for tracking logs, images, and point clouds.

    This logger writes data to MCAP files using Foxglove-compatible schemas,
    allowing visualization in Foxglove Studio and other compatible tools.

    Example:
        >>> with Logger("experiment.mcap") as logger:
        ...     logger.info("Starting training")
        ...     logger.log_image("input", image_bytes, format="png")
        ...     logger.warning("Learning rate might be too high")
    """

    def __init__(
        self,
        output: str | Path | IO[bytes],
        *,
        name: str = "track",
        chunk_size: int = 1024 * 1024,
        compression: str = "zstd",
    ) -> None:
        """Initialize the logger.

        Args:
            output: Output file path or file-like object.
            name: Logger name (included in log messages).
            chunk_size: Size of data chunks in bytes.
            compression: Compression type ('zstd', 'lz4', or 'none').
        """
        self._name = name
        self._output = output
        self._chunk_size = chunk_size
        self._compression = compression

        self._writer: Writer | None = None
        self._file: IO[bytes] | None = None
        self._owns_file = False

        # Channel IDs for different message types
        self._log_channel_id: int | None = None
        self._image_channels: dict[str, int] = {}
        self._pointcloud_channels: dict[str, int] = {}

        # Schema IDs
        self._log_schema_id: int | None = None
        self._image_schema_id: int | None = None
        self._pointcloud_schema_id: int | None = None

        # Sequence counters per channel
        self._sequences: dict[int, int] = {}

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

        # Handle file opening
        if isinstance(self._output, (str, Path)):
            self._file = open(self._output, "wb")
            self._owns_file = True
        else:
            self._file = self._output
            self._owns_file = False

        # Create mcap writer
        from mcap.writer import CompressionType

        compression_map = {
            "zstd": CompressionType.ZSTD,
            "lz4": CompressionType.LZ4,
            "none": CompressionType.NONE,
        }
        compression_type = compression_map.get(self._compression, CompressionType.ZSTD)

        self._writer = Writer(
            self._file,
            chunk_size=self._chunk_size,
            compression=compression_type,
        )
        self._writer.start(profile="", library="track")

    def close(self) -> None:
        """Close the logger and finalize the file."""
        if self._writer is not None:
            self._writer.finish()
            self._writer = None

        if self._owns_file and self._file is not None:
            self._file.close()
            self._file = None

    def _ensure_open(self) -> None:
        """Ensure the logger is open."""
        if self._writer is None:
            raise RuntimeError("Logger is not open. Call open() or use as context manager.")

    def _get_log_channel(self) -> int:
        """Get or create the log channel."""
        if self._log_channel_id is not None:
            return self._log_channel_id

        assert self._writer is not None

        # Register schema if needed
        if self._log_schema_id is None:
            self._log_schema_id = self._writer.register_schema(
                name="foxglove.Log",
                encoding="jsonschema",
                data=get_schema_json(LOG_SCHEMA),
            )

        # Register channel
        self._log_channel_id = self._writer.register_channel(
            topic="/log",
            message_encoding="json",
            schema_id=self._log_schema_id,
        )
        self._sequences[self._log_channel_id] = 0

        return self._log_channel_id

    def _get_image_channel(self, topic: str) -> int:
        """Get or create an image channel for the given topic."""
        if topic in self._image_channels:
            return self._image_channels[topic]

        assert self._writer is not None

        # Register schema if needed
        if self._image_schema_id is None:
            self._image_schema_id = self._writer.register_schema(
                name="foxglove.CompressedImage",
                encoding="jsonschema",
                data=get_schema_json(COMPRESSED_IMAGE_SCHEMA),
            )

        # Register channel
        channel_topic = f"/images/{topic}" if not topic.startswith("/") else topic
        channel_id = self._writer.register_channel(
            topic=channel_topic,
            message_encoding="json",
            schema_id=self._image_schema_id,
        )
        self._image_channels[topic] = channel_id
        self._sequences[channel_id] = 0

        return channel_id

    def _get_pointcloud_channel(self, topic: str) -> int:
        """Get or create a point cloud channel for the given topic."""
        if topic in self._pointcloud_channels:
            return self._pointcloud_channels[topic]

        assert self._writer is not None

        # Register schema if needed
        if self._pointcloud_schema_id is None:
            self._pointcloud_schema_id = self._writer.register_schema(
                name="foxglove.PointCloud",
                encoding="jsonschema",
                data=get_schema_json(POINT_CLOUD_SCHEMA),
            )

        # Register channel
        channel_topic = f"/pointclouds/{topic}" if not topic.startswith("/") else topic
        channel_id = self._writer.register_channel(
            topic=channel_topic,
            message_encoding="json",
            schema_id=self._pointcloud_schema_id,
        )
        self._pointcloud_channels[topic] = channel_id
        self._sequences[channel_id] = 0

        return channel_id

    def _next_sequence(self, channel_id: int) -> int:
        """Get and increment the sequence number for a channel."""
        seq = self._sequences.get(channel_id, 0)
        self._sequences[channel_id] = seq + 1
        return seq

    def _log(
        self,
        level: LogLevel,
        message: str,
        *,
        timestamp_ns: int | None = None,
        file: str | None = None,
        line: int | None = None,
    ) -> None:
        """Internal log method."""
        self._ensure_open()
        assert self._writer is not None

        channel_id = self._get_log_channel()
        ts = timestamp_ns if timestamp_ns is not None else _time_ns()

        # Build log message
        log_msg: dict[str, Any] = {
            "timestamp": _ns_to_timestamp(ts),
            "level": int(level),
            "message": message,
            "name": self._name,
        }

        if file is not None:
            log_msg["file"] = file
        if line is not None:
            log_msg["line"] = line

        # Write message
        data = json.dumps(log_msg).encode("utf-8")
        self._writer.add_message(
            channel_id=channel_id,
            log_time=ts,
            publish_time=ts,
            data=data,
            sequence=self._next_sequence(channel_id),
        )

    def _get_caller_info(self) -> tuple[str | None, int | None]:
        """Get the file and line number of the caller."""
        # Go up the stack to find the actual caller (skip internal methods)
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
        return None, None

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
        data: bytes,
        *,
        format: str = "png",
        frame_id: str = "",
        timestamp_ns: int | None = None,
    ) -> None:
        """Log a compressed image.

        Args:
            topic: Topic name for the image (e.g., "camera/rgb").
            data: Compressed image data (PNG, JPEG, or WebP bytes).
            format: Image format ('png', 'jpeg', or 'webp').
            frame_id: Frame of reference for the image.
            timestamp_ns: Optional timestamp in nanoseconds. Defaults to current time.
        """
        self._ensure_open()
        assert self._writer is not None

        channel_id = self._get_image_channel(topic)
        ts = timestamp_ns if timestamp_ns is not None else _time_ns()

        # Build image message
        image_msg = {
            "timestamp": _ns_to_timestamp(ts),
            "frame_id": frame_id or topic,
            "data": base64.b64encode(data).decode("ascii"),
            "format": format,
        }

        # Write message
        msg_data = json.dumps(image_msg).encode("utf-8")
        self._writer.add_message(
            channel_id=channel_id,
            log_time=ts,
            publish_time=ts,
            data=msg_data,
            sequence=self._next_sequence(channel_id),
        )

    def log_image_array(
        self,
        topic: str,
        array: np.ndarray,
        *,
        format: str = "png",
        frame_id: str = "",
        timestamp_ns: int | None = None,
    ) -> None:
        """Log an image from a numpy array.

        Args:
            topic: Topic name for the image.
            array: Image as numpy array (HxW for grayscale, HxWx3 for RGB, HxWx4 for RGBA).
            format: Output format ('png' or 'jpeg').
            frame_id: Frame of reference for the image.
            timestamp_ns: Optional timestamp in nanoseconds. Defaults to current time.

        Note:
            Requires PIL/Pillow to be installed for encoding.
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "PIL/Pillow is required for log_image_array. "
                "Install with: pip install Pillow"
            )

        # Convert array to PIL Image
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
        data = buffer.getvalue()

        self.log_image(
            topic=topic,
            data=data,
            format=format,
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
        )

    def log_pointcloud(
        self,
        topic: str,
        points: np.ndarray,
        *,
        colors: np.ndarray | None = None,
        intensities: np.ndarray | None = None,
        frame_id: str = "world",
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        timestamp_ns: int | None = None,
    ) -> None:
        """Log a point cloud.

        Args:
            topic: Topic name for the point cloud.
            points: Nx3 array of point positions (x, y, z).
            colors: Optional Nx3 or Nx4 array of RGB or RGBA colors (0-255).
            intensities: Optional N array of intensity values.
            frame_id: Frame of reference.
            position: Origin position (x, y, z).
            orientation: Origin orientation as quaternion (x, y, z, w).
            timestamp_ns: Optional timestamp in nanoseconds. Defaults to current time.
        """
        self._ensure_open()
        assert self._writer is not None

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be Nx3 array, got shape {points.shape}")

        n_points = points.shape[0]

        # Define fields and calculate stride
        fields: list[dict[str, Any]] = []
        offset = 0

        # XYZ coordinates (float32)
        for name in ["x", "y", "z"]:
            fields.append({"name": name, "offset": offset, "type": NumericType.FLOAT32})
            offset += 4

        # Optional intensity (float32)
        has_intensity = intensities is not None
        if has_intensity:
            fields.append({"name": "intensity", "offset": offset, "type": NumericType.FLOAT32})
            offset += 4

        # Optional color (RGBA as uint8)
        has_color = colors is not None
        if has_color:
            for name in ["red", "green", "blue", "alpha"]:
                fields.append({"name": name, "offset": offset, "type": NumericType.UINT8})
                offset += 1

        point_stride = offset

        # Pack point data
        data_buffer = bytearray(n_points * point_stride)
        points_f32 = points.astype(np.float32)

        for i in range(n_points):
            base = i * point_stride
            # XYZ
            struct.pack_into("fff", data_buffer, base, *points_f32[i])
            field_offset = 12

            # Intensity
            if has_intensity:
                struct.pack_into("f", data_buffer, base + field_offset, float(intensities[i]))
                field_offset += 4

            # Color
            if has_color:
                if colors.shape[1] == 3:
                    # RGB, add full alpha
                    r, g, b = colors[i].astype(np.uint8)
                    struct.pack_into("BBBB", data_buffer, base + field_offset, r, g, b, 255)
                else:
                    # RGBA
                    r, g, b, a = colors[i].astype(np.uint8)
                    struct.pack_into("BBBB", data_buffer, base + field_offset, r, g, b, a)

        channel_id = self._get_pointcloud_channel(topic)
        ts = timestamp_ns if timestamp_ns is not None else _time_ns()

        # Build point cloud message
        pc_msg = {
            "timestamp": _ns_to_timestamp(ts),
            "frame_id": frame_id,
            "pose": {
                "position": {"x": position[0], "y": position[1], "z": position[2]},
                "orientation": {
                    "x": orientation[0],
                    "y": orientation[1],
                    "z": orientation[2],
                    "w": orientation[3],
                },
            },
            "point_stride": point_stride,
            "fields": fields,
            "data": base64.b64encode(bytes(data_buffer)).decode("ascii"),
        }

        # Write message
        msg_data = json.dumps(pc_msg).encode("utf-8")
        self._writer.add_message(
            channel_id=channel_id,
            log_time=ts,
            publish_time=ts,
            data=msg_data,
            sequence=self._next_sequence(channel_id),
        )

    def add_metadata(self, name: str, data: dict[str, str]) -> None:
        """Add metadata to the MCAP file.

        Args:
            name: Metadata name/category.
            data: Key-value metadata pairs.
        """
        self._ensure_open()
        assert self._writer is not None
        self._writer.add_metadata(name=name, data=data)

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
        self._writer.add_attachment(
            create_time=ts,
            log_time=ts,
            name=name,
            media_type=media_type,
            data=data,
        )
