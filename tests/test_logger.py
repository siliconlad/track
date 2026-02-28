"""Tests for Logger behavior."""

import ast
import inspect
import io
import os
import queue
import re
import threading
from pathlib import Path

import numpy as np
import pytest

import track
from track import LogLevel, Logger
from track import logger as logger_module
from track.logger import LogRecord, RecordType
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


class _NoopProcess:
    """Process test double for close-path unit tests."""

    def __init__(self) -> None:
        self.join_calls: list[float] = []
        self.terminated = False

    def join(self, timeout: float = 0.0) -> None:
        self.join_calls.append(timeout)

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        self.terminated = True


def _logger_calls(path: Path) -> list[ast.Call]:
    """Collect direct Logger(...) call nodes from an example file."""
    module = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Logger"
    ]


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
        expected_dir = tmp_path / "env_output"
        monkeypatch.setenv("TRACK_OUTPUT_DIR", str(expected_dir))
        with Logger("env_logger") as logger:
            logger.info("Test message")
            output_path = logger.output_path

        assert output_path.exists()
        assert output_path.parent == expected_dir
        assert output_path.name.endswith("_env_logger.mcap")

    def test_default_output_dir_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no output override is provided, default path should be used."""
        monkeypatch.delenv("TRACK_OUTPUT_DIR", raising=False)
        logger = Logger("default_logger")
        expected_dir = Path.home() / ".local" / "track" / "logs"
        assert logger.output_path.parent == expected_dir
        assert logger.output_path.name.endswith("_default_logger.mcap")

    def test_default_output_open_creates_parent_dirs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Opening with default path should create missing parent directories."""
        monkeypatch.delenv("TRACK_OUTPUT_DIR", raising=False)
        monkeypatch.setattr(logger_module.Path, "home", classmethod(lambda cls: tmp_path))

        with Logger("default_logger") as logger:
            logger.info("hello from default output")
            output_path = logger.output_path

        expected_output_dir = tmp_path / ".local" / "track" / "logs"
        assert expected_output_dir.exists()
        assert output_path.parent == expected_output_dir

        with McapFileReader.from_file(output_path) as reader:
            messages = list(reader.messages("/log"))

        assert len(messages) == 1
        assert messages[0].data.msg == "hello from default output"

    def test_output_dir_creates_timestamped_file_name(self, tmp_path: Path) -> None:
        """Directory output should generate a timestamped file based on logger name."""
        output_dir = tmp_path / "runs"

        with Logger("train job", output_dir=output_dir) as logger:
            logger.info("Test message")
            output_path = logger.output_path

        assert output_path.exists()
        assert output_path.parent == output_dir
        assert re.match(r"^\d{8}_\d{6}_\d{6}_train_job\.mcap$", output_path.name)

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

    def test_process_mode_close_enqueues_close_record(self, tmp_path: Path) -> None:
        """Process-mode close should enqueue a CLOSE sentinel record."""
        output_file = tmp_path / "close_queue.mcap"
        logger = Logger("test", output_dir=output_file, use_process=True)
        capture = _CaptureQueue()
        writer_process = _NoopProcess()
        logger._queue = capture
        logger._writer_process = writer_process
        logger._owner_pid = os.getpid()

        logger.close()

        assert len(capture.records) == 1
        close_record = capture.records[0]
        assert close_record.record_type is RecordType.CLOSE
        assert close_record.data == {}
        assert writer_process.join_calls == [10.0]

    def test_log_caller_metadata_points_to_immediate_callsite(self, tmp_path: Path) -> None:
        """Caller metadata should reference the line where logger.info is called."""
        output_file = tmp_path / "caller_metadata.mcap"
        with Logger("test", output_dir=output_file) as logger:
            frame = inspect.currentframe()
            assert frame is not None
            expected_line = frame.f_lineno + 1
            logger.info("caller metadata")

        with McapFileReader.from_file(output_file) as reader:
            messages = list(reader.messages("/log"))

        assert len(messages) == 1
        logged = messages[0].data
        assert logged.file
        assert Path(logged.file).name == Path(__file__).name
        assert logged.line == expected_line


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_level_values(self) -> None:
        """LogLevel values are stable."""
        assert LogLevel.DEBUG == 10
        assert LogLevel.INFO == 20
        assert LogLevel.WARNING == 30
        assert LogLevel.ERROR == 40
        assert LogLevel.FATAL == 50


@pytest.mark.parametrize(
    ("example_relpath", "logger_name"),
    [
        ("examples/basic_usage.py", "example"),
        ("examples/threaded_logging.py", "threaded"),
        ("examples/multiprocess_logging.py", "multiprocess"),
    ],
)
def test_examples_use_valid_logger_constructor(example_relpath: str, logger_name: str) -> None:
    """Examples should call Logger(name, output_dir=...) with valid arguments."""
    example_path = Path(__file__).resolve().parent.parent / example_relpath
    calls = _logger_calls(example_path)

    assert calls, f"Expected at least one Logger(...) call in {example_relpath}"

    call = calls[0]
    assert call.args, f"Expected positional logger name argument in {example_relpath}"
    assert isinstance(call.args[0], ast.Constant)
    assert call.args[0].value == logger_name

    keyword_names = {kw.arg for kw in call.keywords if kw.arg is not None}
    assert "output_dir" in keyword_names
    assert "name" not in keyword_names


@pytest.fixture(autouse=False)
def _clean_registry():
    """Reset global registry state after each test that uses it."""
    yield
    # Teardown: close any remaining default logger and clear registry
    try:
        track.finish()
    except Exception:
        pass
    with track._registry_lock:
        # Force-close any remaining loggers
        for logger in list(track._loggers.values()):
            try:
                logger.close()
            except Exception:
                pass
        track._loggers.clear()
        track._default_logger = None


class TestRegistry:
    """Tests for the global logger registry."""

    @pytest.fixture(autouse=True)
    def _setup(self, _clean_registry):
        """Apply clean_registry fixture to all tests in this class."""

    def test_get_logger_returns_registered_logger(self, tmp_path: Path) -> None:
        """init() registers a logger retrievable via get_logger()."""
        output_dir = tmp_path / "registry"
        logger = track.init("mylogger", output_dir=output_dir)
        retrieved = track.get_logger("mylogger")
        assert retrieved is logger

    def test_get_logger_unknown_name_raises(self) -> None:
        """get_logger() raises KeyError for unregistered names."""
        with pytest.raises(KeyError, match="no_such_logger"):
            track.get_logger("no_such_logger")

    def test_init_duplicate_name_raises(self, tmp_path: Path) -> None:
        """init() raises ValueError when a logger with the same name exists."""
        track.init("dup", output_dir=tmp_path / "dup1")
        with pytest.raises(ValueError, match="already registered"):
            track.init("dup", output_dir=tmp_path / "dup2")

    def test_finish_unregisters_logger(self, tmp_path: Path) -> None:
        """After finish(), the logger is no longer in the registry."""
        track.init("temp", output_dir=tmp_path / "temp")
        track.finish()
        with pytest.raises(KeyError):
            track.get_logger("temp")

    def test_finish_when_no_default_is_noop(self) -> None:
        """Calling finish() without init() does not error."""
        track.finish()  # should not raise


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def _setup(self, _clean_registry):
        """Apply clean_registry fixture to all tests in this class."""

    def test_module_level_info_writes_message(self, tmp_path: Path) -> None:
        """track.info() writes a message through the default logger."""
        output_dir = tmp_path / "conv"
        logger = track.init("conv", output_dir=output_dir)
        track.info("hello from module level")
        track.finish()

        with McapFileReader.from_file(logger.output_path) as reader:
            messages = list(reader.messages("/log"))

        assert len(messages) == 1
        assert messages[0].data.msg == "hello from module level"
        assert messages[0].data.name == "conv"
        assert messages[0].data.level == int(LogLevel.INFO)

    def test_module_level_all_log_levels(self, tmp_path: Path) -> None:
        """All five convenience log functions write correctly."""
        output_dir = tmp_path / "levels"
        logger = track.init("levels", output_dir=output_dir)
        track.debug("d")
        track.info("i")
        track.warning("w")
        track.error("e")
        track.fatal("f")
        track.finish()

        with McapFileReader.from_file(logger.output_path) as reader:
            messages = list(reader.messages("/log"))

        assert len(messages) == 5
        assert [m.data.level for m in messages] == [
            int(LogLevel.DEBUG),
            int(LogLevel.INFO),
            int(LogLevel.WARNING),
            int(LogLevel.ERROR),
            int(LogLevel.FATAL),
        ]

    def test_module_level_log_image(self, tmp_path: Path) -> None:
        """track.log_image() writes an image through the default logger."""
        output_dir = tmp_path / "img"
        logger = track.init("img", output_dir=output_dir)
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        track.log_image("cam", image, format="png")
        track.finish()

        with McapFileReader.from_file(logger.output_path) as reader:
            messages = list(reader.messages("/images/cam"))

        assert len(messages) == 1

    def test_module_level_caller_metadata(self, tmp_path: Path) -> None:
        """Caller metadata should point to the actual callsite, not the wrapper."""
        output_dir = tmp_path / "caller"
        logger = track.init("caller", output_dir=output_dir)
        frame = inspect.currentframe()
        assert frame is not None
        expected_line = frame.f_lineno + 1
        track.info("caller test")
        track.finish()

        with McapFileReader.from_file(logger.output_path) as reader:
            messages = list(reader.messages("/log"))

        assert len(messages) == 1
        logged = messages[0].data
        assert logged.file
        assert Path(logged.file).name == Path(__file__).name
        assert logged.line == expected_line

    def test_convenience_without_init_raises(self) -> None:
        """Calling convenience functions before init() raises RuntimeError."""
        with pytest.raises(RuntimeError, match="track.init"):
            track.info("should fail")
