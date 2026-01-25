"""Tests for concurrent logging utilities."""

import multiprocessing as mp
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from track import AsyncLogger


class TestAsyncLogger:
    """Tests for AsyncLogger."""

    def test_single_thread_with_thread_backend(self, tmp_path: Path) -> None:
        """Test basic operation with thread backend."""
        output_file = tmp_path / "test.mcap"

        with AsyncLogger(output_file, use_process=False) as logger:
            logger.info("Test message")
            logger.debug("Debug message")
            logger.warning("Warning message")
            time.sleep(0.2)  # Allow queue to drain

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_multiple_threads(self, tmp_path: Path) -> None:
        """Test logging from multiple threads."""
        output_file = tmp_path / "test.mcap"
        num_threads = 4
        messages_per_thread = 50

        def worker(logger: AsyncLogger, thread_id: int) -> None:
            for i in range(messages_per_thread):
                logger.info(f"Thread {thread_id}: message {i}")

        with AsyncLogger(output_file) as logger:
            threads = [
                threading.Thread(target=worker, args=(logger, i)) for i in range(num_threads)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            time.sleep(0.5)  # Allow queue to drain

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_pointcloud_logging(self, tmp_path: Path) -> None:
        """Test logging point clouds with structured array."""
        output_file = tmp_path / "test.mcap"

        # Create structured array with XYZ and RGB
        dtype = np.dtype([
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ])
        points = np.zeros(100, dtype=dtype)
        points["x"] = np.random.randn(100)
        points["y"] = np.random.randn(100)
        points["z"] = np.random.randn(100)
        points["r"] = np.random.randint(0, 255, 100)
        points["g"] = np.random.randint(0, 255, 100)
        points["b"] = np.random.randint(0, 255, 100)

        with AsyncLogger(output_file) as logger:
            logger.log_pointcloud("test_pc", points)
            time.sleep(0.2)

        assert output_file.exists()

    def test_metadata(self, tmp_path: Path) -> None:
        """Test adding metadata."""
        output_file = tmp_path / "test.mcap"

        with AsyncLogger(output_file) as logger:
            logger.add_metadata("experiment", {"name": "test", "version": "1.0"})
            time.sleep(0.2)

        assert output_file.exists()

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test that context manager properly opens and closes."""
        output_file = tmp_path / "test.mcap"

        with AsyncLogger(output_file) as logger:
            logger.info("Test")
            time.sleep(0.1)

        assert output_file.exists()

    def test_not_open_error(self) -> None:
        """Test that operations on closed logger raise errors."""
        logger = AsyncLogger("test.mcap")

        with pytest.raises(RuntimeError, match="not open"):
            logger.info("Test")


class TestAsyncLoggerMultiProcess:
    """Tests for AsyncLogger with process backend."""

    @pytest.fixture(autouse=True)
    def setup_spawn(self) -> None:
        """Ensure spawn method for compatibility."""
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    def test_single_process_with_process_backend(self, tmp_path: Path) -> None:
        """Test basic operation with process backend."""
        output_file = tmp_path / "test.mcap"

        with AsyncLogger(output_file, use_process=True) as logger:
            logger.info("Test message")
            logger.debug("Debug message")
            time.sleep(0.2)

        assert output_file.exists()
