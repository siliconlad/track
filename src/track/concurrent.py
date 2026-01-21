"""Concurrent logging support for multi-threaded and multi-process environments.

This module provides thread-safe and process-safe logging utilities:

- ThreadSafeLogger: Logger with locking for multi-threaded applications
- MultiProcessLogger: Queue-based logger for multi-process applications
- merge_mcap_files: Utility to merge multiple MCAP files

Architecture Overview:

## Thread Safety (ThreadSafeLogger)
Uses a reentrant lock (RLock) to protect all write operations. Safe to use
from multiple threads writing to the same logger instance.

## Process Safety (MultiProcessLogger)
Uses a queue-based architecture where:
- Worker processes serialize log records to a multiprocessing.Queue
- A dedicated writer process/thread consumes from the queue and writes to MCAP
- This avoids file corruption from concurrent writes

## Distributed Systems
For distributed training across multiple machines:
- Each machine uses its own Logger with a unique filename
- Use merge_mcap_files() to combine results after training
- Or use per-rank filenames: experiment_rank0.mcap, experiment_rank1.mcap
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

from track.logger import Logger, LogLevel, _time_ns


class RecordType(Enum):
    """Types of log records for queue-based logging."""

    LOG = auto()
    IMAGE = auto()
    IMAGE_ARRAY = auto()
    POINTCLOUD = auto()
    METADATA = auto()
    ATTACHMENT = auto()
    CLOSE = auto()


@dataclass
class LogRecord:
    """A log record to be written by the writer process."""

    record_type: RecordType
    timestamp_ns: int
    data: dict[str, Any]


class ThreadSafeLogger(Logger):
    """Thread-safe version of Logger using locks.

    All write operations are protected by a reentrant lock, making it safe
    to use from multiple threads simultaneously.

    Example:
        >>> logger = ThreadSafeLogger("experiment.mcap")
        >>> logger.open()
        >>>
        >>> def worker(worker_id):
        ...     for i in range(100):
        ...         logger.info(f"Worker {worker_id}: iteration {i}")
        >>>
        >>> threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        >>> for t in threads:
        ...     t.start()
        >>> for t in threads:
        ...     t.join()
        >>>
        >>> logger.close()
    """

    def __init__(
        self,
        output: str | Path,
        *,
        name: str = "track",
        chunk_size: int = 1024 * 1024,
        compression: str = "zstd",
    ) -> None:
        """Initialize thread-safe logger.

        Args:
            output: Output file path.
            name: Logger name.
            chunk_size: Size of data chunks in bytes.
            compression: Compression type ('zstd', 'lz4', or 'none').
        """
        super().__init__(
            output,
            name=name,
            chunk_size=chunk_size,
            compression=compression,
        )
        self._lock = threading.RLock()

    def open(self) -> None:
        """Open the logger for writing (thread-safe)."""
        with self._lock:
            super().open()

    def close(self) -> None:
        """Close the logger (thread-safe)."""
        with self._lock:
            super().close()

    def _log(
        self,
        level: LogLevel,
        message: str,
        *,
        timestamp_ns: int | None = None,
        file: str | None = None,
        line: int | None = None,
    ) -> None:
        """Internal log method (thread-safe)."""
        with self._lock:
            super()._log(level, message, timestamp_ns=timestamp_ns, file=file, line=line)

    def log_image(
        self,
        topic: str,
        data: bytes,
        *,
        format: str = "png",
        frame_id: str = "",
        timestamp_ns: int | None = None,
    ) -> None:
        """Log a compressed image (thread-safe)."""
        with self._lock:
            super().log_image(
                topic, data, format=format, frame_id=frame_id, timestamp_ns=timestamp_ns
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
        """Log an image from numpy array (thread-safe)."""
        with self._lock:
            super().log_image_array(
                topic, array, format=format, frame_id=frame_id, timestamp_ns=timestamp_ns
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
        """Log a point cloud (thread-safe)."""
        with self._lock:
            super().log_pointcloud(
                topic,
                points,
                colors=colors,
                intensities=intensities,
                frame_id=frame_id,
                position=position,
                orientation=orientation,
                timestamp_ns=timestamp_ns,
            )

    def add_metadata(self, name: str, data: dict[str, str]) -> None:
        """Add metadata (thread-safe)."""
        with self._lock:
            super().add_metadata(name, data)

    def add_attachment(
        self,
        name: str,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        timestamp_ns: int | None = None,
    ) -> None:
        """Add an attachment (thread-safe)."""
        with self._lock:
            super().add_attachment(name, data, media_type=media_type, timestamp_ns=timestamp_ns)


def _writer_process(
    output_path: str,
    record_queue: mp.Queue,
    name: str,
    chunk_size: int,
    compression: str,
    ready_event: mp.Event,
) -> None:
    """Writer process that consumes records from queue and writes to MCAP.

    This runs in a separate process to handle all file I/O.
    """
    logger = Logger(
        output_path,
        name=name,
        chunk_size=chunk_size,
        compression=compression,
    )
    logger.open()

    # Signal that we're ready to receive records
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
                # Log errors but continue processing
                print(f"Error processing record: {e}")

    finally:
        logger.close()


def _process_record(logger: Logger, record: LogRecord) -> None:
    """Process a single log record."""
    data = record.data

    if record.record_type == RecordType.LOG:
        # Use internal _log to bypass caller info detection
        logger._ensure_open()
        assert logger._writer is not None

        channel_id = logger._get_log_channel()
        ts = record.timestamp_ns

        import json

        from track.logger import _ns_to_timestamp

        log_msg = {
            "timestamp": _ns_to_timestamp(ts),
            "level": data["level"],
            "message": data["message"],
            "name": data["name"],
        }
        if data.get("file"):
            log_msg["file"] = data["file"]
        if data.get("line"):
            log_msg["line"] = data["line"]

        msg_data = json.dumps(log_msg).encode("utf-8")
        logger._writer.add_message(
            channel_id=channel_id,
            log_time=ts,
            publish_time=ts,
            data=msg_data,
            sequence=logger._next_sequence(channel_id),
        )

    elif record.record_type == RecordType.IMAGE:
        logger.log_image(
            topic=data["topic"],
            data=data["data"],
            format=data["format"],
            frame_id=data["frame_id"],
            timestamp_ns=record.timestamp_ns,
        )

    elif record.record_type == RecordType.IMAGE_ARRAY:
        logger.log_image_array(
            topic=data["topic"],
            array=data["array"],
            format=data["format"],
            frame_id=data["frame_id"],
            timestamp_ns=record.timestamp_ns,
        )

    elif record.record_type == RecordType.POINTCLOUD:
        logger.log_pointcloud(
            topic=data["topic"],
            points=data["points"],
            colors=data.get("colors"),
            intensities=data.get("intensities"),
            frame_id=data["frame_id"],
            position=data["position"],
            orientation=data["orientation"],
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


class MultiProcessLogger:
    """Process-safe logger using a queue-based architecture.

    This logger is safe to use from multiple processes. It uses a dedicated
    writer process that consumes log records from a multiprocessing queue.

    Example:
        >>> from multiprocessing import Process
        >>>
        >>> def worker(logger, worker_id):
        ...     for i in range(100):
        ...         logger.info(f"Worker {worker_id}: iteration {i}")
        >>>
        >>> with MultiProcessLogger("experiment.mcap") as logger:
        ...     processes = [
        ...         Process(target=worker, args=(logger, i))
        ...         for i in range(4)
        ...     ]
        ...     for p in processes:
        ...         p.start()
        ...     for p in processes:
        ...         p.join()

    Note:
        The logger must be created in the main process before spawning workers.
        Pass the logger instance to worker processes as an argument.
    """

    def __init__(
        self,
        output: str | Path,
        *,
        name: str = "track",
        chunk_size: int = 1024 * 1024,
        compression: str = "zstd",
        queue_size: int = 10000,
    ) -> None:
        """Initialize multi-process logger.

        Args:
            output: Output file path.
            name: Logger name.
            chunk_size: Size of data chunks in bytes.
            compression: Compression type ('zstd', 'lz4', or 'none').
            queue_size: Maximum number of records to buffer in queue.
        """
        self._output = str(output)
        self._name = name
        self._chunk_size = chunk_size
        self._compression = compression
        self._queue_size = queue_size

        self._queue: mp.Queue | None = None
        self._writer_process: mp.Process | None = None
        self._ready_event: mp.Event | None = None
        self._closed = False

    def __enter__(self) -> MultiProcessLogger:
        """Enter context manager."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.close()

    def open(self) -> None:
        """Start the writer process."""
        if self._writer_process is not None:
            return

        # Create queue and synchronization event
        self._queue = mp.Queue(maxsize=self._queue_size)
        self._ready_event = mp.Event()

        # Start writer process
        self._writer_process = mp.Process(
            target=_writer_process,
            args=(
                self._output,
                self._queue,
                self._name,
                self._chunk_size,
                self._compression,
                self._ready_event,
            ),
            daemon=False,  # Ensure proper cleanup
        )
        self._writer_process.start()

        # Wait for writer to be ready
        self._ready_event.wait(timeout=10.0)

        # Register cleanup on exit
        atexit.register(self.close)

    def close(self) -> None:
        """Stop the writer process and finalize the file."""
        if self._closed:
            return
        self._closed = True

        if self._queue is not None and self._writer_process is not None:
            # Send close signal
            try:
                self._queue.put(LogRecord(RecordType.CLOSE, 0, {}), timeout=5.0)
            except queue.Full:
                pass

            # Wait for writer to finish
            self._writer_process.join(timeout=10.0)

            if self._writer_process.is_alive():
                self._writer_process.terminate()
                self._writer_process.join(timeout=1.0)

        self._queue = None
        self._writer_process = None

        # Unregister atexit handler
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    def _ensure_open(self) -> None:
        """Ensure the logger is open."""
        if self._queue is None:
            raise RuntimeError(
                "MultiProcessLogger is not open. Call open() or use as context manager."
            )

    def _put_record(self, record: LogRecord, timeout: float = 1.0) -> None:
        """Put a record on the queue."""
        self._ensure_open()
        assert self._queue is not None
        try:
            self._queue.put(record, timeout=timeout)
        except queue.Full:
            # Drop record if queue is full (log warning to stderr)
            import sys

            print(f"Warning: Log queue full, dropping record", file=sys.stderr)

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

    def _get_caller_info(self) -> tuple[str | None, int | None]:
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
        return None, None

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

    def log_image(
        self,
        topic: str,
        data: bytes,
        *,
        format: str = "png",
        frame_id: str = "",
        timestamp_ns: int | None = None,
    ) -> None:
        """Log a compressed image."""
        ts = timestamp_ns if timestamp_ns is not None else _time_ns()
        record = LogRecord(
            RecordType.IMAGE,
            ts,
            {
                "topic": topic,
                "data": data,
                "format": format,
                "frame_id": frame_id,
            },
        )
        self._put_record(record)

    def log_image_array(
        self,
        topic: str,
        array: np.ndarray,
        *,
        format: str = "png",
        frame_id: str = "",
        timestamp_ns: int | None = None,
    ) -> None:
        """Log an image from numpy array."""
        ts = timestamp_ns if timestamp_ns is not None else _time_ns()
        # Make a copy to avoid issues with shared memory
        record = LogRecord(
            RecordType.IMAGE_ARRAY,
            ts,
            {
                "topic": topic,
                "array": array.copy(),
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
        colors: np.ndarray | None = None,
        intensities: np.ndarray | None = None,
        frame_id: str = "world",
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        timestamp_ns: int | None = None,
    ) -> None:
        """Log a point cloud."""
        ts = timestamp_ns if timestamp_ns is not None else _time_ns()
        record = LogRecord(
            RecordType.POINTCLOUD,
            ts,
            {
                "topic": topic,
                "points": points.copy(),
                "colors": colors.copy() if colors is not None else None,
                "intensities": intensities.copy() if intensities is not None else None,
                "frame_id": frame_id,
                "position": position,
                "orientation": orientation,
            },
        )
        self._put_record(record)

    def add_metadata(self, name: str, data: dict[str, str]) -> None:
        """Add metadata."""
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
        """Add an attachment."""
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


class ProcessLocalLogger:
    """Factory for creating process-local loggers in distributed settings.

    Each process gets its own MCAP file, avoiding any synchronization overhead.
    Use merge_mcap_files() to combine results after all processes complete.

    Example:
        >>> # In distributed training script
        >>> factory = ProcessLocalLogger(
        ...     output_dir="logs",
        ...     base_name="experiment",
        ... )
        >>>
        >>> # Each process gets its own logger
        >>> logger = factory.get_logger()  # Creates experiment_rank0.mcap, etc.
        >>> logger.info("Training started")
        >>>
        >>> # After training, merge all files
        >>> factory.merge_all("experiment_combined.mcap")
    """

    def __init__(
        self,
        output_dir: str | Path,
        base_name: str = "experiment",
        *,
        rank: int | None = None,
        world_size: int | None = None,
        chunk_size: int = 1024 * 1024,
        compression: str = "zstd",
    ) -> None:
        """Initialize process-local logger factory.

        Args:
            output_dir: Directory for output files.
            base_name: Base name for output files.
            rank: Process rank (auto-detected from env vars if not provided).
            world_size: Total number of processes (auto-detected if not provided).
            chunk_size: Size of data chunks in bytes.
            compression: Compression type.
        """
        self._output_dir = Path(output_dir)
        self._base_name = base_name
        self._chunk_size = chunk_size
        self._compression = compression

        # Auto-detect rank and world_size from common environment variables
        if rank is None:
            rank = self._detect_rank()
        if world_size is None:
            world_size = self._detect_world_size()

        self._rank = rank
        self._world_size = world_size
        self._logger: Logger | None = None

    def _detect_rank(self) -> int:
        """Detect process rank from environment variables."""
        # Try common distributed training env vars
        for var in ["RANK", "LOCAL_RANK", "SLURM_PROCID", "PMI_RANK"]:
            if var in os.environ:
                return int(os.environ[var])
        return 0

    def _detect_world_size(self) -> int:
        """Detect world size from environment variables."""
        for var in ["WORLD_SIZE", "SLURM_NTASKS", "PMI_SIZE"]:
            if var in os.environ:
                return int(os.environ[var])
        return 1

    @property
    def rank(self) -> int:
        """Get the process rank."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Get the world size."""
        return self._world_size

    @property
    def output_path(self) -> Path:
        """Get the output path for this process."""
        return self._output_dir / f"{self._base_name}_rank{self._rank}.mcap"

    def get_logger(self) -> Logger:
        """Get or create a logger for this process.

        Returns:
            A Logger instance writing to a process-specific file.
        """
        if self._logger is None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._logger = Logger(
                self.output_path,
                name=f"{self._base_name}_rank{self._rank}",
                chunk_size=self._chunk_size,
                compression=self._compression,
            )
            self._logger.open()

            # Add rank info as metadata
            self._logger.add_metadata(
                "distributed",
                {
                    "rank": str(self._rank),
                    "world_size": str(self._world_size),
                },
            )

        return self._logger

    def close(self) -> None:
        """Close the logger if open."""
        if self._logger is not None:
            self._logger.close()
            self._logger = None

    def get_all_output_paths(self) -> list[Path]:
        """Get output paths for all ranks."""
        return [
            self._output_dir / f"{self._base_name}_rank{r}.mcap"
            for r in range(self._world_size)
        ]

    def merge_all(self, output_path: str | Path) -> None:
        """Merge all rank files into a single file.

        This should only be called after all processes have finished.

        Args:
            output_path: Path for the merged output file.
        """
        input_paths = [p for p in self.get_all_output_paths() if p.exists()]
        merge_mcap_files(input_paths, output_path)


def merge_mcap_files(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    sort_by_time: bool = True,
) -> None:
    """Merge multiple MCAP files into a single file.

    This is useful for combining logs from distributed training or
    parallel processing where each process wrote to its own file.

    Args:
        input_paths: List of input MCAP file paths.
        output_path: Output file path for merged result.
        sort_by_time: Whether to sort messages by timestamp.

    Example:
        >>> merge_mcap_files(
        ...     ["rank0.mcap", "rank1.mcap", "rank2.mcap"],
        ...     "combined.mcap"
        ... )
    """
    from mcap.reader import make_reader
    from mcap.writer import Writer

    # Collect all messages with their metadata
    messages: list[tuple[int, int, bytes, str, str, bytes]] = []
    schemas: dict[str, tuple[str, bytes]] = {}  # name -> (encoding, data)
    channels: dict[str, tuple[str, str, str]] = {}  # topic -> (encoding, schema_name, topic)
    all_metadata: list[tuple[str, dict[str, str]]] = []
    all_attachments: list[tuple[int, int, str, str, bytes]] = []

    for input_path in input_paths:
        with open(input_path, "rb") as f:
            reader = make_reader(f)

            # Get summary for schemas and channels
            summary = reader.get_summary()

            if summary is not None:
                # Collect schemas
                for schema_id, schema in summary.schemas.items():
                    if schema.name not in schemas:
                        schemas[schema.name] = (schema.encoding, schema.data)

                # Collect channels
                for channel_id, channel in summary.channels.items():
                    schema_name = ""
                    if channel.schema_id in summary.schemas:
                        schema_name = summary.schemas[channel.schema_id].name
                    if channel.topic not in channels:
                        channels[channel.topic] = (
                            channel.message_encoding,
                            schema_name,
                            channel.topic,
                        )

            # Collect messages
            for schema, channel, message in reader.iter_messages():
                schema_name = schema.name if schema else ""
                messages.append(
                    (
                        message.log_time,
                        message.publish_time,
                        message.data,
                        channel.topic,
                        schema_name,
                        channel.message_encoding.encode(),
                    )
                )

    # Sort by log_time if requested
    if sort_by_time:
        messages.sort(key=lambda x: x[0])

    # Write merged file
    with open(output_path, "wb") as f:
        writer = Writer(f)
        writer.start(profile="", library="track-merge")

        # Register schemas
        schema_ids: dict[str, int] = {}
        for name, (encoding, data) in schemas.items():
            schema_id = writer.register_schema(name=name, encoding=encoding, data=data)
            schema_ids[name] = schema_id

        # Register channels
        channel_ids: dict[str, int] = {}
        for topic, (msg_encoding, schema_name, _) in channels.items():
            schema_id = schema_ids.get(schema_name, 0)
            channel_id = writer.register_channel(
                topic=topic,
                message_encoding=msg_encoding,
                schema_id=schema_id,
            )
            channel_ids[topic] = channel_id

        # Write messages
        sequence_counters: dict[int, int] = {}
        for log_time, publish_time, data, topic, schema_name, msg_encoding in messages:
            channel_id = channel_ids.get(topic)
            if channel_id is None:
                continue

            seq = sequence_counters.get(channel_id, 0)
            sequence_counters[channel_id] = seq + 1

            writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                publish_time=publish_time,
                data=data,
                sequence=seq,
            )

        writer.finish()
