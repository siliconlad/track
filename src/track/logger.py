"""ML experiment logger built on pybag."""

from __future__ import annotations

import atexit
import inspect
import io
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Any

import numpy as np
import pybag.types as t
from PIL import Image
from pybag.mcap_writer import McapFileWriter
from pybag.ros2.humble import builtin_interfaces, sensor_msgs, std_msgs


class LogLevel(IntEnum):
    """Logging level."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    FATAL = 50


# TODO: Use the rcl_interfaces message in pybag when added
@dataclass(kw_only=True)
class Log:
    __msg_name__ = "rcl_interfaces/msg/Log"

    stamp: builtin_interfaces.Time
    """Timestamp associated with the log message."""

    level: t.uint8
    """Logging level."""

    name: t.string
    """Name of logger that this message came from."""

    msg: t.string
    """Full log message."""

    file: t.string
    """File the log message came from."""

    line: t.uint32
    """Line number in the file the log message came from."""


class RecordType(Enum):
    """Types of records for process-queue logging."""

    LOG = auto()
    IMAGE = auto()
    POINTCLOUD = auto()
    METADATA = auto()
    ATTACHMENT = auto()
    CLOSE = auto()


@dataclass
class LogRecord:
    """A write request for the background writer process."""

    record_type: RecordType
    data: dict[str, Any]


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


def _now_ns() -> int:
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


def _process_record(logger: Logger, record: LogRecord) -> None:
    """Process one queued record in the writer process."""
    data = record.data

    if record.record_type == RecordType.LOG:
        logger._log(
            LogLevel(data["level"]),
            data["message"],
            file=data["file"],
            line=data["line"],
            timestamp_ns=data["timestamp"],
        )
    elif record.record_type == RecordType.IMAGE:
        logger.log_image(
            topic=data["topic"],
            image=data["image"],
            format=data["format"],
            frame_id=data["frame_id"],
            timestamp_ns=data["timestamp"],
        )
    elif record.record_type == RecordType.POINTCLOUD:
        logger.log_pointcloud(
            topic=data["topic"],
            points=data["points"],
            frame_id=data["frame_id"],
            timestamp_ns=data["timestamp"],
        )
    elif record.record_type == RecordType.METADATA:
        logger.add_metadata(data["name"], data["data"])
    elif record.record_type == RecordType.ATTACHMENT:
        logger.add_attachment(
            data["name"],
            data["data"],
            media_type=data["media_type"],
            timestamp_ns=data["timestamp"],
        )


def _writer_process_main(
    output_path: str,
    record_queue: mp.Queue,
    name: str,
    ready_event: mp.Event,
) -> None:
    """Writer process target that drains queue records into MCAP."""
    logger = Logger(name, output_path, use_process=False)

    try:
        logger.open()
        ready_event.set()

        while True:
            try:
                record: LogRecord = record_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if record.record_type == RecordType.CLOSE:
                break

            try:
                _process_record(logger, record)
            except Exception as exc:
                print(f"Error processing log record: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"Error starting writer process: {exc}", file=sys.stderr)
        ready_event.set()
    finally:
        logger.close()


class Logger:
    """Logger for tracking logs, images, and point clouds.

    The default mode writes synchronously and is safe to share across threads.
    For multi-process producers, pass `use_process=True` to enqueue records and
    write from a dedicated background process.
    """

    def __init__(
        self,
        name: str,
        output_dir: str | Path | None = None,
        *,
        use_process: bool = False,
    ) -> None:
        """Initialize the logger.

        Args:
            output: Output MCAP file path.
            name: Logger name (included in log messages).
            use_process: If True, enqueue writes and use a writer process.
        """
        self._name = name
        self._use_process = use_process
        self._queue_size = 1000  # TODO: Expose as param?
        self._output = self._resolve_output(output_dir)

        self._writer: McapFileWriter | None = None
        self._lock = threading.Lock()
        self._closed = False

        self._queue: mp.Queue | None = None
        self._writer_process: mp.Process | None = None
        self._owner_pid: int | None = None
        self._atexit_registered = False

    def __enter__(self) -> Logger:
        """Enter context manager."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        """Support pickling when sharing logger to spawned child processes."""
        state = self.__dict__.copy()
        state["_lock"] = None
        state["_writer"] = None
        state["_writer_process"] = None
        state["_atexit_registered"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickled logger state."""
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._writer = None
        self._writer_process = None
        self._atexit_registered = False

    def _resolve_output(self, output_dir_override: str | Path | None) -> Path:
        """Determines the output directory for the log file"""
        if output_dir_override is not None:
            return Path(output_dir_override)
        if os.environ.get("TRACK_OUTPUT_DIR") is not None:
            return Path(os.environ["TRACK_OUTPUT_DIR"])
        # Default fallback in case nothing is specified
        return Path.home() / ".local" / "track" / "logs"

    def open(self) -> None:
        """Open the logger for writing."""
        with self._lock:
            if self._is_open():
                return

            self._closed = False
            self._owner_pid = os.getpid()
            self._open_writer()

            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True

    def _is_open(self) -> bool:
        """Check whether logger resources are initialized."""
        if self._use_process:
            return self._queue is not None
        return self._writer is not None

    def _open_writer(self):
        """Set up the writer for this logger."""
        if self._use_process:
            # Open separate writer process
            self._open_writer_process()
        else:
            # Open writer in this process
            self._writer = McapFileWriter.open(self._output, mode="w", profile="ros2")

    def _open_writer_process(self) -> None:
        """Open process queue mode. Lock must be held."""
        self._queue = mp.Queue(maxsize=self._queue_size)
        ready_event = mp.Event()

        self._writer_process = mp.Process(
            target=_writer_process_main,
            args=(self._output, self._queue, self._name, ready_event),
            daemon=False,
        )

        try:
            self._writer_process.start()
            if not ready_event.wait(timeout=10.0):
                raise RuntimeError("Writer process failed to start.")
            if not self._writer_process.is_alive():
                exitcode = self._writer_process.exitcode
                raise RuntimeError(f"Writer process exited unexpectedly (exitcode={exitcode}).")
        except Exception:
            # Cleanup writer process
            if self._writer_process is not None:
                if self._writer_process.is_alive():
                    self._writer_process.terminate()
                self._writer_process.join(timeout=1.0)
                self._writer_process = None
            # Cleanup queue
            if self._queue is not None:
                self._queue.close()
                self._queue.join_thread()
                self._queue = None

            raise

    def close(self) -> None:
        """Close the logger and finalize the file."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

            writer = self._writer
            record_queue = self._queue
            writer_process = self._writer_process
            owner_pid = self._owner_pid
            use_process = self._use_process

            self._writer = None
            self._queue = None
            self._writer_process = None
            self._owner_pid = None

            if self._atexit_registered:
                try:
                    atexit.unregister(self.close)
                except Exception:
                    pass
                self._atexit_registered = False

        if not use_process:
            if writer is not None:
                writer.close()
            return

        # In process mode, only the owner process should control writer shutdown.
        if owner_pid is not None and os.getpid() != owner_pid:
            return

        if record_queue is not None:
            try:
                record_queue.put(LogRecord(RecordType.CLOSE, 0, {}), timeout=5.0)
            except (queue.Full, Exception):
                pass

        if writer_process is not None:
            writer_process.join(timeout=10.0)
            if writer_process.is_alive():
                writer_process.terminate()
                writer_process.join(timeout=1.0)

        if record_queue is not None:
            try:
                record_queue.close()
                record_queue.join_thread()
            except Exception:
                pass

    def _get_caller_info(self) -> tuple[str, int]:
        """Get the file and line number of the caller."""
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
        file: str,
        line: int,
        *,
        timestamp_ns: int | None = None,
    ) -> None:
        """Internal log method."""
        ts = _now_ns() if timestamp_ns is None else timestamp_ns

        # Multiprocess implementation
        if self._use_process:
            self._put_record(
                LogRecord(
                    RecordType.LOG,
                    {
                        "level": int(level),
                        "message": message,
                        "file": file,
                        "line": line,
                        "timestamp": ts,
                    },
                )
            )
            return

        # Single process implementation
        with self._lock:
            self._write_log(level, message, file=file, line=line, timestamp_ns=ts)

    def _put_record(self, record: LogRecord, timeout: float = 1.0) -> None:
        """Enqueue one record for the background writer process."""
        try:
            assert self._queue is not None
            self._queue.put(record, timeout=timeout)
        except queue.Full:
            # TODO: Is there something better we can do?
            warnings.warn("Log queue full, dropping record!", stacklevel=2)

    def _write_log(
        self,
        level: LogLevel,
        message: str,
        *,
        file: str,
        line: int,
        timestamp_ns: int,
    ) -> None:
        """Write a log entry directly. Lock should be held."""
        assert self._writer is not None, "Open the logger first with open()"
        log_msg = Log(
            stamp=_ns_to_stamp(timestamp_ns),
            level=int(level),
            name=self._name,
            msg=message,
            file=file,
            line=line,
        )
        self._writer.write_message("/log", timestamp_ns, log_msg)

    def debug(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log a debug message."""
        file, line = self._get_caller_info()
        self._log(LogLevel.DEBUG, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def info(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log an info message."""
        file, line = self._get_caller_info()
        self._log(LogLevel.INFO, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def warning(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log a warning message."""
        file, line = self._get_caller_info()
        self._log(LogLevel.WARNING, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def error(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log an error message."""
        file, line = self._get_caller_info()
        self._log(LogLevel.ERROR, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def fatal(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log a fatal message."""
        file, line = self._get_caller_info()
        self._log(LogLevel.FATAL, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def add_metadata(self, name: str, data: dict[str, str]) -> None:
        """Add metadata."""
        # Multiprocess implementation
        if self._use_process:
            self._put_record(
                LogRecord(
                    RecordType.METADATA,
                    {"name": name, "data": data.copy()},
                )
            )
            return
        # Single process implementation
        with self._lock:
            assert self._writer is not None, "Open the logger first with open()"
            self._writer.write_metadata(name, data)

    def add_attachment(
        self,
        name: str,
        data: bytes,
        media_type: str = "application/octet-stream",
        *,
        timestamp_ns: int | None = None,
    ) -> None:
        """Add an attachment.

        Attachments are useful for storing auxiliary data like model weights,
        configuration files, or other binary data.
        """
        ts = timestamp_ns if timestamp_ns is not None else _now_ns()

        # Multiprocess implementation
        if self._use_process:
            self._put_record(
                LogRecord(
                    RecordType.ATTACHMENT,
                    {
                        "name": name,
                        "data": data,
                        "media_type": media_type,
                        "timestamp": ts,
                    },
                )
            )
            return

        # Single process implementation
        with self._lock:
            assert self._writer is not None, "Open the logger first with open()"
            self._writer.write_attachment(name, data, media_type=media_type, log_time=ts)

    def log_image(
        self,
        topic: str,
        image: bytes | np.ndarray,
        *,
        format: str = "png",
        frame_id: str | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        """Log an image.

        Args:
            topic: Topic name (e.g., "camera/rgb").
            image: Image data as compressed bytes or numpy array.
            format: Image format ('png', 'jpeg', or 'webp').
            frame_id: Frame of reference for the image.
            timestamp_ns: Optional timestamp in nanoseconds.
        """
        ts = timestamp_ns if timestamp_ns is not None else _now_ns()
        image_format = format.lower()

        # Multiprocess implementation
        if self._use_process:
            image_data = image.copy() if isinstance(image, np.ndarray) else image
            self._put_record(
                LogRecord(
                    RecordType.IMAGE,
                    {
                        "topic": topic,
                        "image": image_data,
                        "format": image_format,
                        "frame_id": frame_id or "",
                        "timestamp": ts,
                    },
                )
            )
            return

        # Single process implementation
        with self._lock:
            self._write_image(
                topic=topic,
                image=image,
                format=image_format,
                frame_id=frame_id or "",
                timestamp_ns=ts,
            )

    def _write_image(
        self,
        *,
        topic: str,
        image: bytes | np.ndarray,
        format: str,
        frame_id: str,
        timestamp_ns: int,
    ) -> None:
        """Write image directly. Lock should be held."""
        assert self._writer is not None, "Open the logger first with open()"
        channel_topic = f"/images/{topic}" if not topic.startswith("/") else topic

        if isinstance(image, np.ndarray):
            image = self._encode_image_array(image, format)

        msg = sensor_msgs.CompressedImage(
            header=_make_header(timestamp_ns, frame_id or topic),
            format=format,
            data=list(image),
        )
        self._writer.write_message(channel_topic, timestamp_ns, msg)

    def _encode_image_array(self, array: np.ndarray, format: str) -> bytes:
        """Encode a numpy array to compressed image bytes."""
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
        """Log a point cloud."""
        ts = timestamp_ns if timestamp_ns is not None else _now_ns()

        # Multiprocess implementation
        if self._use_process:
            self._put_record(
                LogRecord(
                    RecordType.POINTCLOUD,
                    {
                        "topic": topic,
                        "points": points.copy(),
                        "frame_id": frame_id,
                        "timestamp": ts,
                    },
                )
            )
            return

        # Single process implementation
        with self._lock:
            self._write_pointcloud(topic=topic, points=points, frame_id=frame_id, timestamp_ns=ts)

    def _write_pointcloud(
        self,
        *,
        topic: str,
        points: np.ndarray,
        frame_id: str,
        timestamp_ns: int,
    ) -> None:
        """Write point cloud directly. Lock should be held."""
        assert self._writer is not None, "Open logger first with open()"
        channel_topic = f"/pointclouds/{topic}" if not topic.startswith("/") else topic

        if not isinstance(points.dtype, np.dtype) or points.dtype.names is None:
            raise ValueError(
                "points must be a numpy structured array (recarray). "
                "Example: np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])"
            )

        fields = []
        for name in points.dtype.names:
            field_dtype = points.dtype.fields[name][0]
            offset = points.dtype.fields[name][1]

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

        n_points = len(points)
        point_step = points.dtype.itemsize
        row_step = n_points * point_step
        data = points.tobytes()

        msg = sensor_msgs.PointCloud2(
            header=_make_header(timestamp_ns, frame_id),
            height=1,
            width=n_points,
            fields=fields,
            is_bigendian=False,
            point_step=point_step,
            row_step=row_step,
            data=list(data),
            is_dense=True,
        )
        self._writer.write_message(channel_topic, timestamp_ns, msg)
