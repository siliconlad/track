"""Concurrent logging support for multi-threaded and multi-process environments.

This module provides non-blocking, async logging utilities:

- AsyncLogger: Queue-based logger with background writer (works for threads and processes)

## AsyncLogger Architecture

The AsyncLogger uses a queue-based architecture where:
- Log calls immediately enqueue records and return (non-blocking)
- A background writer (thread or process) consumes records and writes to MCAP
- This decouples logging from I/O, improving application performance

The logger can use either:
- A background thread (default, lower overhead, suitable for multi-threaded apps)
- A background process (for multi-process apps where the logger is shared)
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

from track.logger import Logger, LogLevel


def _time_ns() -> int:
    """Get current time in nanoseconds."""
    return time.time_ns()


class RecordType(Enum):
    """Types of log records for queue-based logging."""

    LOG = auto()
    IMAGE = auto()
    POINTCLOUD = auto()
    METADATA = auto()
    ATTACHMENT = auto()
    CLOSE = auto()


@dataclass
class LogRecord:
    """A log record to be written by the background writer."""

    record_type: RecordType
    timestamp_ns: int
    data: dict[str, Any]


def _process_record(logger: Logger, record: LogRecord) -> None:
    """Process a single log record."""
    data = record.data

    if record.record_type == RecordType.LOG:
        logger._log(
            LogLevel(data["level"]),
            data["message"],
            timestamp_ns=record.timestamp_ns,
            file=data.get("file", ""),
            line=data.get("line", 0),
        )

    elif record.record_type == RecordType.IMAGE:
        logger.log_image(
            topic=data["topic"],
            image=data["image"],
            format=data["format"],
            frame_id=data["frame_id"],
            timestamp_ns=record.timestamp_ns,
        )

    elif record.record_type == RecordType.POINTCLOUD:
        logger.log_pointcloud(
            topic=data["topic"],
            points=data["points"],
            frame_id=data["frame_id"],
            timestamp_ns=record.timestamp_ns,
        )

    elif record.record_type == RecordType.METADATA:
        logger.add_metadata(data["name"], data["data"])

    elif record.record_type == RecordType.ATTACHMENT:
        logger.add_attachment(
            data["name"],
            data["data"],
            media_type=data["media_type"],
            timestamp_ns=record.timestamp_ns,
        )


def _writer_thread_main(
    output_path: str,
    record_queue: queue.Queue,
    name: str,
    chunk_size: int,
    compression: str,
    ready_event: threading.Event,
) -> None:
    """Writer thread that consumes records from queue and writes to MCAP."""
    logger = Logger(
        output_path,
        name=name,
        chunk_size=chunk_size,
        compression=compression,
    )
    logger.open()
    ready_event.set()

    try:
        while True:
            try:
                record: LogRecord = record_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if record.record_type == RecordType.CLOSE:
                break

            try:
                _process_record(logger, record)
            except Exception as e:
                import sys

                print(f"Error processing log record: {e}", file=sys.stderr)
    finally:
        logger.close()


def _writer_process_main(
    output_path: str,
    record_queue: mp.Queue,
    name: str,
    chunk_size: int,
    compression: str,
    ready_event: mp.Event,
) -> None:
    """Writer process that consumes records from queue and writes to MCAP."""
    logger = Logger(
        output_path,
        name=name,
        chunk_size=chunk_size,
        compression=compression,
    )
    logger.open()
    ready_event.set()

    try:
        while True:
            try:
                record: LogRecord = record_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if record.record_type == RecordType.CLOSE:
                break

            try:
                _process_record(logger, record)
            except Exception as e:
                import sys

                print(f"Error processing log record: {e}", file=sys.stderr)
    finally:
        logger.close()


class AsyncLogger:
    """Non-blocking logger with background writer.

    Log calls immediately enqueue records and return, while a background
    writer handles all file I/O. This is ideal for performance-critical
    applications where you don't want logging to block your main loop.

    Works for both multi-threaded and multi-process applications:
    - Use `use_process=False` (default) for multi-threaded apps
    - Use `use_process=True` for multi-process apps

    Example (multi-threaded):
        >>> with AsyncLogger("experiment.mcap") as logger:
        ...     def worker(worker_id):
        ...         for i in range(100):
        ...             logger.info(f"Worker {worker_id}: iteration {i}")
        ...
        ...     threads = [Thread(target=worker, args=(i,)) for i in range(4)]
        ...     for t in threads:
        ...         t.start()
        ...     for t in threads:
        ...         t.join()

    Example (multi-process):
        >>> with AsyncLogger("experiment.mcap", use_process=True) as logger:
        ...     def worker(logger, worker_id):
        ...         for i in range(100):
        ...             logger.info(f"Worker {worker_id}: iteration {i}")
        ...
        ...     processes = [Process(target=worker, args=(logger, i)) for i in range(4)]
        ...     for p in processes:
        ...         p.start()
        ...     for p in processes:
        ...         p.join()
    """

    def __init__(
        self,
        output: str | Path,
        *,
        name: str = "track",
        chunk_size: int = 1024 * 1024,
        compression: str = "lz4",
        queue_size: int = 10000,
        use_process: bool = False,
    ) -> None:
        """Initialize async logger.

        Args:
            output: Output file path.
            name: Logger name (included in log messages).
            chunk_size: Size of data chunks in bytes.
            compression: Compression type ('zstd', 'lz4', or 'none').
            queue_size: Maximum number of records to buffer in queue.
            use_process: If True, use a background process (for multi-process apps).
                        If False (default), use a background thread.
        """
        self._output = str(output)
        self._name = name
        self._chunk_size = chunk_size
        self._compression = compression
        self._queue_size = queue_size
        self._use_process = use_process

        self._queue: queue.Queue | mp.Queue | None = None
        self._writer_thread: threading.Thread | None = None
        self._writer_process: mp.Process | None = None
        self._ready_event: threading.Event | mp.Event | None = None
        self._closed = False

    def __enter__(self) -> AsyncLogger:
        """Enter context manager."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.close()

    def open(self) -> None:
        """Start the background writer."""
        if self._queue is not None:
            return

        if self._use_process:
            self._open_with_process()
        else:
            self._open_with_thread()

        atexit.register(self.close)

    def _open_with_thread(self) -> None:
        """Start background writer thread."""
        self._queue = queue.Queue(maxsize=self._queue_size)
        self._ready_event = threading.Event()

        self._writer_thread = threading.Thread(
            target=_writer_thread_main,
            args=(
                self._output,
                self._queue,
                self._name,
                self._chunk_size,
                self._compression,
                self._ready_event,
            ),
            daemon=True,
        )
        self._writer_thread.start()
        self._ready_event.wait(timeout=10.0)

    def _open_with_process(self) -> None:
        """Start background writer process."""
        self._queue = mp.Queue(maxsize=self._queue_size)
        self._ready_event = mp.Event()

        self._writer_process = mp.Process(
            target=_writer_process_main,
            args=(
                self._output,
                self._queue,
                self._name,
                self._chunk_size,
                self._compression,
                self._ready_event,
            ),
            daemon=False,
        )
        self._writer_process.start()
        self._ready_event.wait(timeout=10.0)

    def close(self) -> None:
        """Stop the background writer and finalize the file."""
        if self._closed:
            return
        self._closed = True

        if self._queue is not None:
            # Send close signal
            try:
                self._queue.put(LogRecord(RecordType.CLOSE, 0, {}), timeout=5.0)
            except (queue.Full, Exception):
                pass

            # Wait for writer to finish
            if self._writer_thread is not None:
                self._writer_thread.join(timeout=10.0)
                self._writer_thread = None

            if self._writer_process is not None:
                self._writer_process.join(timeout=10.0)
                if self._writer_process.is_alive():
                    self._writer_process.terminate()
                    self._writer_process.join(timeout=1.0)
                self._writer_process = None

        self._queue = None
        self._ready_event = None

        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    def _ensure_open(self) -> None:
        """Ensure the logger is open."""
        if self._queue is None:
            raise RuntimeError(
                "AsyncLogger is not open. Call open() or use as context manager."
            )

    def _put_record(self, record: LogRecord, timeout: float = 1.0) -> None:
        """Put a record on the queue (non-blocking if queue has space)."""
        self._ensure_open()
        assert self._queue is not None
        try:
            self._queue.put(record, timeout=timeout)
        except queue.Full:
            import sys

            print("Warning: Log queue full, dropping record", file=sys.stderr)

    def _get_caller_info(self) -> tuple[str, int]:
        """Get the file and line number of the caller."""
        import inspect

        frame = inspect.currentframe()
        try:
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
        ts = timestamp_ns if timestamp_ns is not None else _time_ns()
        record = LogRecord(
            RecordType.LOG,
            ts,
            {
                "level": int(level),
                "message": message,
                "name": self._name,
                "file": file,
                "line": line,
            },
        )
        self._put_record(record)

    def debug(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log a debug message (non-blocking)."""
        file, line = self._get_caller_info()
        self._log(LogLevel.DEBUG, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def info(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log an info message (non-blocking)."""
        file, line = self._get_caller_info()
        self._log(LogLevel.INFO, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def warning(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log a warning message (non-blocking)."""
        file, line = self._get_caller_info()
        self._log(LogLevel.WARNING, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def error(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log an error message (non-blocking)."""
        file, line = self._get_caller_info()
        self._log(LogLevel.ERROR, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def fatal(self, message: str, *, timestamp_ns: int | None = None) -> None:
        """Log a fatal message (non-blocking)."""
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
        """Log an image (non-blocking).

        Args:
            topic: Topic name for the image.
            image: Image data - either compressed bytes or numpy array.
            format: Image format ('png', 'jpeg', or 'webp').
            frame_id: Frame of reference for the image.
            timestamp_ns: Optional timestamp in nanoseconds.
        """
        ts = timestamp_ns if timestamp_ns is not None else _time_ns()
        # Copy array to avoid issues with shared memory
        image_data = image.copy() if isinstance(image, np.ndarray) else image
        record = LogRecord(
            RecordType.IMAGE,
            ts,
            {
                "topic": topic,
                "image": image_data,
                "format": format,
                "frame_id": frame_id,
            },
        )
        self._put_record(record)

    def log_pointcloud(
        self,
        topic: str,
        points: np.ndarray,
        *,
        frame_id: str = "world",
        timestamp_ns: int | None = None,
    ) -> None:
        """Log a point cloud (non-blocking).

        Args:
            topic: Topic name for the point cloud.
            points: Point cloud as numpy structured array.
            frame_id: Frame of reference.
            timestamp_ns: Optional timestamp in nanoseconds.
        """
        ts = timestamp_ns if timestamp_ns is not None else _time_ns()
        record = LogRecord(
            RecordType.POINTCLOUD,
            ts,
            {
                "topic": topic,
                "points": points.copy(),
                "frame_id": frame_id,
            },
        )
        self._put_record(record)

    def add_metadata(self, name: str, data: dict[str, str]) -> None:
        """Add metadata (non-blocking)."""
        record = LogRecord(RecordType.METADATA, 0, {"name": name, "data": data.copy()})
        self._put_record(record)

    def add_attachment(
        self,
        name: str,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        timestamp_ns: int | None = None,
    ) -> None:
        """Add an attachment (non-blocking)."""
        ts = timestamp_ns if timestamp_ns is not None else _time_ns()
        record = LogRecord(
            RecordType.ATTACHMENT,
            ts,
            {
                "name": name,
                "data": data,
                "media_type": media_type,
            },
        )
        self._put_record(record)
