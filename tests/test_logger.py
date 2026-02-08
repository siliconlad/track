"""Tests for Logger behavior."""

import io
import queue
import threading
from pathlib import Path

import numpy as np
import pytest

from track import LogLevel, Logger
from track.logger import LogRecord
from pybag.mcap_reader import McapFileReader
from PIL import Image


def _thread_worker(logger: Logger, thread_id: int, messages_per_thread: int) -> None:
    """Write messages from a worker thread."""
    for i in range(messages_per_thread):
        logger.info(f"Thread {thread_id}: message {i}")


class _CaptureQueue:
    """In-memory queue test double for process-mode unit tests."""

    def __init__(self) -> None:
        self.records: list[LogRecord] = []

    def put(self, record: LogRecord, timeout: float = 1.0) -> None:  # noqa: ARG002
        self.records.append(record)


class _FullQueue:
    """Queue test double that always raises queue.Full."""

    def put(self, record: LogRecord, timeout: float = 1.0) -> None:  # noqa: ARG002
        raise queue.Full


class TestLogger:
    """Tests for logger behavior."""

    def test_context_manager(self, tmp_path: Path) -> None:
        """Logger can be used as a context manager."""
        output_file = tmp_path / "context.mcap"
        with Logger("test", output_dir=output_file) as logger:
            logger.info("Test message")

        with McapFileReader.from_file(output_file) as reader:
            messages = list(reader.messages("/log"))

        assert len(messages) == 1
        assert messages[0].data.msg == "Test message"
        assert messages[0].data.name == "test"
        assert messages[0].data.level == int(LogLevel.INFO)

    def test_not_open_error(self, tmp_path: Path) -> None:
        """Calling write methods before open triggers an assertion."""
        output_file = tmp_path / "not_open.mcap"
        logger = Logger("test", output_dir=output_file)
        with pytest.raises(AssertionError, match=r"open\(\)"):
            logger.info("Test")

    def test_output_dir_from_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """TRACK_OUTPUT_DIR should be used when output_dir is omitted."""
        expected_path = tmp_path / "env_output.mcap"
        monkeypatch.setenv("TRACK_OUTPUT_DIR", str(expected_path))
        with Logger("env_logger") as logger:
            logger.info("Test message")
        assert expected_path.exists()

    def test_default_output_dir_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no output override is provided, default path should be used."""
        monkeypatch.delenv("TRACK_OUTPUT_DIR", raising=False)
        logger = Logger("default_logger")
        assert logger._output == Path.home() / ".local" / "track" / "logs"

    def test_log_levels(self, tmp_path: Path) -> None:
        """All log methods write without errors."""
        output_file = tmp_path / "levels.mcap"
        with Logger("levels", output_dir=output_file) as logger:
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.fatal("Fatal message")

        with McapFileReader.from_file(output_file) as reader:
            messages = list(reader.messages("/log"))

        assert len(messages) == 5
        assert [msg.data.level for msg in messages] == [
            int(LogLevel.DEBUG),
            int(LogLevel.INFO),
            int(LogLevel.WARNING),
            int(LogLevel.ERROR),
            int(LogLevel.FATAL),
        ]

    def test_log_image_bytes(self, tmp_path: Path) -> None:
        """Log compressed image bytes."""
        output_file = tmp_path / "image_bytes.mcap"
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, :, 0] = 255

        # Convert to bytes
        buffer = io.BytesIO()
        Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
        png_data = buffer.getvalue()

        with Logger("test", output_dir=output_file) as logger:
            logger.log_image("test_image", png_data, format="png")

        with McapFileReader.from_file(output_file) as reader:
            messages = list(reader.messages("/images/test_image"))

        assert len(messages) == 1
        image_msg = messages[0].data
        assert image_msg.format == "png"
        assert image_msg.data == png_data

    def test_log_image_array(self, tmp_path: Path) -> None:
        """Log a numpy image array."""
        pil_image = pytest.importorskip("PIL.Image")
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :, 0] = 255

        output_file = tmp_path / "image_array.mcap"
        with Logger("test", output_dir=output_file) as logger:
            logger.log_image("test_image", image, format="png")

        with McapFileReader.from_file(output_file) as reader:
            messages = list(reader.messages("/images/test_image"))

        assert len(messages) == 1
        image_msg = messages[0].data
        decoded = np.array(pil_image.open(io.BytesIO(image_msg.data)))
        assert decoded.shape == (64, 64, 3)
        assert np.all(decoded[:, :, 0] == 255)
        assert np.all(decoded[:, :, 1] == 0)
        assert np.all(decoded[:, :, 2] == 0)

    def test_log_pointcloud_with_intensity(self, tmp_path: Path) -> None:
        """Log a structured point cloud with intensity field."""
        output_file = tmp_path / "pointcloud_intensity.mcap"

        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4")])
        points = np.zeros(3, dtype=dtype)
        points["x"] = [0.0, 1.0, 0.0]
        points["y"] = [0.0, 0.0, 1.0]
        points["z"] = [0.0, 0.0, 0.0]
        points["intensity"] = [0.5, 0.8, 1.0]

        with Logger("test", output_dir=output_file) as logger:
            logger.log_pointcloud("test_pc", points)

        with McapFileReader.from_file(output_file) as reader:
            messages = list(reader.messages("/pointclouds/test_pc"))

        assert len(messages) == 1
        pointcloud = messages[0].data
        logged_points = np.frombuffer(pointcloud.data, dtype=dtype, count=3)
        np.testing.assert_allclose(logged_points["x"], points["x"])
        np.testing.assert_allclose(logged_points["y"], points["y"])
        np.testing.assert_allclose(logged_points["z"], points["z"])
        np.testing.assert_allclose(logged_points["intensity"], points["intensity"])

    def test_invalid_pointcloud_not_structured(self, tmp_path: Path) -> None:
        """Non-structured arrays raise validation errors."""
        output_file = tmp_path / "bad_pointcloud.mcap"
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        with Logger("test", output_dir=output_file) as logger:
            with pytest.raises(ValueError, match="structured array"):
                logger.log_pointcloud("test_pc", points)

    def test_add_metadata(self, tmp_path: Path) -> None:
        """Write metadata section."""
        output_file = tmp_path / "metadata.mcap"
        with Logger("test", output_dir=output_file) as logger:
            logger.add_metadata("experiment", {"name": "test", "version": "1.0"})

        with McapFileReader.from_file(output_file) as reader:
            metadata = reader.get_metadata("experiment")

        assert len(metadata) == 1
        assert metadata[0].metadata == {"name": "test", "version": "1.0"}

    def test_add_attachment(self, tmp_path: Path) -> None:
        """Write attachment payload."""
        output_file = tmp_path / "attachment.mcap"
        payload = b'{"key": "value"}'
        with Logger("test", output_dir=output_file) as logger:
            logger.add_attachment(
                "config.json",
                payload,
                media_type="application/json",
            )

        with McapFileReader.from_file(output_file) as reader:
            attachments = reader.get_attachments("config.json")

        assert len(attachments) == 1
        assert attachments[0].media_type == "application/json"
        assert attachments[0].data == payload

    def test_multiple_threads_direct_mode(self, tmp_path: Path) -> None:
        """Logger can be shared across threads in direct mode."""
        output_file = tmp_path / "threaded.mcap"
        num_threads = 4
        messages_per_thread = 50

        with Logger("test", output_dir=output_file) as logger:
            threads = [
                threading.Thread(target=_thread_worker, args=(logger, i, messages_per_thread))
                for i in range(num_threads)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        with McapFileReader.from_file(output_file) as reader:
            messages = list(reader.messages("/log"))

        assert len(messages) == num_threads * messages_per_thread

    def test_process_mode_log_image_copies_array_before_enqueue(self, tmp_path: Path) -> None:
        """Process mode should enqueue a copy of ndarray image payloads."""
        output_file = tmp_path / "image_queue.mcap"
        logger = Logger("test", output_dir=output_file, use_process=True)
        capture = _CaptureQueue()
        logger._queue = capture

        image = np.zeros((2, 2, 3), dtype=np.uint8)
        logger.log_image("camera", image, format="png")
        image[:, :, :] = 255

        assert len(capture.records) == 1
        queued_image = capture.records[0].data["image"]
        assert isinstance(queued_image, np.ndarray)
        assert np.all(queued_image == 0)

    def test_process_mode_log_pointcloud_copies_array_before_enqueue(self, tmp_path: Path) -> None:
        """Process mode should enqueue a copy of pointcloud arrays."""
        output_file = tmp_path / "pointcloud_queue.mcap"
        logger = Logger("test", output_dir=output_file, use_process=True)
        capture = _CaptureQueue()
        logger._queue = capture

        points = np.zeros(2, dtype=np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")]))
        logger.log_pointcloud("lidar", points)
        points["x"][:] = 9.0

        assert len(capture.records) == 1
        queued_points = capture.records[0].data["points"]
        assert isinstance(queued_points, np.ndarray)
        assert np.all(queued_points["x"] == 0.0)

    def test_process_mode_warns_when_queue_is_full(self, tmp_path: Path) -> None:
        """When queue is full in process mode, records should be dropped with warning."""
        output_file = tmp_path / "full_queue.mcap"
        logger = Logger("test", output_dir=output_file, use_process=True)
        logger._queue = _FullQueue()

        with pytest.warns(UserWarning, match="queue full"):
            logger.info("drop me")


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_level_values(self) -> None:
        """LogLevel values are stable."""
        assert LogLevel.DEBUG == 10
        assert LogLevel.INFO == 20
        assert LogLevel.WARNING == 30
        assert LogLevel.ERROR == 40
        assert LogLevel.FATAL == 50
