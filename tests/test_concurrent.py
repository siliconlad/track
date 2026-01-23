"""Tests for concurrent logging utilities."""

import multiprocessing as mp
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from track import AsyncLogger, merge_mcap_files


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
        """Test logging point clouds."""
        output_file = tmp_path / "test.mcap"

        with AsyncLogger(output_file) as logger:
            points = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
            logger.log_pointcloud("test_pc", points, colors=colors)
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
        import io

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


class TestMergeMcapFiles:
    """Tests for merge_mcap_files."""

    def test_merge_two_files(self, tmp_path: Path) -> None:
        """Test merging two MCAP files."""
        file1 = tmp_path / "file1.mcap"
        file2 = tmp_path / "file2.mcap"
        merged = tmp_path / "merged.mcap"

        # Create first file
        with AsyncLogger(file1, name="file1") as logger:
            logger.info("Message from file 1")
            time.sleep(0.1)

        # Create second file
        with AsyncLogger(file2, name="file2") as logger:
            logger.info("Message from file 2")
            time.sleep(0.1)

        # Merge
        merge_mcap_files([file1, file2], merged)

        assert merged.exists()
        assert merged.stat().st_size > 0

    def test_merge_with_sorting(self, tmp_path: Path) -> None:
        """Test that merged files are sorted by time."""
        file1 = tmp_path / "file1.mcap"
        file2 = tmp_path / "file2.mcap"
        merged = tmp_path / "merged.mcap"

        # Create files with specific timestamps
        ts1 = 1000000000  # 1 second
        ts2 = 500000000  # 0.5 seconds (earlier)

        with AsyncLogger(file1) as logger:
            logger.info("Later message", timestamp_ns=ts1)
            time.sleep(0.1)

        with AsyncLogger(file2) as logger:
            logger.info("Earlier message", timestamp_ns=ts2)
            time.sleep(0.1)

        # Merge with sorting
        merge_mcap_files([file1, file2], merged, sort_by_time=True)

        assert merged.exists()

    def test_merge_empty_list(self, tmp_path: Path) -> None:
        """Test merging empty list of files."""
        merged = tmp_path / "merged.mcap"

        merge_mcap_files([], merged)

        assert merged.exists()
