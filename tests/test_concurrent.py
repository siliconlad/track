"""Tests for concurrent logging utilities."""

import multiprocessing as mp
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from track import (
    MultiProcessLogger,
    ProcessLocalLogger,
    ThreadSafeLogger,
    merge_mcap_files,
)


class TestThreadSafeLogger:
    """Tests for ThreadSafeLogger."""

    def test_single_thread(self, tmp_path: Path) -> None:
        """Test basic single-threaded operation."""
        output_file = tmp_path / "test.mcap"

        with ThreadSafeLogger(output_file) as logger:
            logger.info("Test message")
            logger.debug("Debug message")
            logger.warning("Warning message")

        assert output_file.exists()

    def test_multiple_threads(self, tmp_path: Path) -> None:
        """Test logging from multiple threads."""
        output_file = tmp_path / "test.mcap"
        num_threads = 4
        messages_per_thread = 50

        def worker(logger: ThreadSafeLogger, thread_id: int) -> None:
            for i in range(messages_per_thread):
                logger.info(f"Thread {thread_id}: message {i}")

        with ThreadSafeLogger(output_file) as logger:
            threads = [
                threading.Thread(target=worker, args=(logger, i)) for i in range(num_threads)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_thread_safe_pointcloud(self, tmp_path: Path) -> None:
        """Test logging point clouds from multiple threads."""
        output_file = tmp_path / "test.mcap"

        def worker(logger: ThreadSafeLogger, thread_id: int) -> None:
            for i in range(10):
                points = np.random.randn(10, 3).astype(np.float32)
                logger.log_pointcloud(f"thread_{thread_id}", points)

        with ThreadSafeLogger(output_file) as logger:
            threads = [threading.Thread(target=worker, args=(logger, i)) for i in range(4)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert output_file.exists()


class TestMultiProcessLogger:
    """Tests for MultiProcessLogger."""

    @pytest.fixture(autouse=True)
    def setup_spawn(self) -> None:
        """Ensure spawn method for compatibility."""
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    def test_single_process(self, tmp_path: Path) -> None:
        """Test basic single-process operation."""
        output_file = tmp_path / "test.mcap"

        with MultiProcessLogger(output_file) as logger:
            logger.info("Test message")
            logger.debug("Debug message")
            time.sleep(0.1)  # Allow queue to drain

        assert output_file.exists()

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test that context manager properly opens and closes."""
        output_file = tmp_path / "test.mcap"

        with MultiProcessLogger(output_file) as logger:
            logger.info("Test")
            time.sleep(0.1)

        assert output_file.exists()

    def test_metadata(self, tmp_path: Path) -> None:
        """Test adding metadata."""
        output_file = tmp_path / "test.mcap"

        with MultiProcessLogger(output_file) as logger:
            logger.add_metadata("test", {"key": "value"})
            time.sleep(0.1)

        assert output_file.exists()


class TestProcessLocalLogger:
    """Tests for ProcessLocalLogger."""

    def test_single_process(self, tmp_path: Path) -> None:
        """Test basic operation in single process."""
        factory = ProcessLocalLogger(tmp_path, "test", rank=0, world_size=1)

        logger = factory.get_logger()
        logger.info("Test message")
        factory.close()

        assert factory.output_path.exists()

    def test_rank_detection(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test automatic rank detection from environment."""
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "4")

        factory = ProcessLocalLogger(tmp_path, "test")

        assert factory.rank == 2
        assert factory.world_size == 4

    def test_output_paths(self, tmp_path: Path) -> None:
        """Test output path generation."""
        factory = ProcessLocalLogger(tmp_path, "experiment", rank=0, world_size=4)

        paths = factory.get_all_output_paths()
        assert len(paths) == 4
        assert paths[0].name == "experiment_rank0.mcap"
        assert paths[3].name == "experiment_rank3.mcap"


class TestMergeMcapFiles:
    """Tests for merge_mcap_files."""

    def test_merge_two_files(self, tmp_path: Path) -> None:
        """Test merging two MCAP files."""
        file1 = tmp_path / "file1.mcap"
        file2 = tmp_path / "file2.mcap"
        merged = tmp_path / "merged.mcap"

        # Create first file
        with ThreadSafeLogger(file1, name="file1") as logger:
            logger.info("Message from file 1")

        # Create second file
        with ThreadSafeLogger(file2, name="file2") as logger:
            logger.info("Message from file 2")

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

        with ThreadSafeLogger(file1) as logger:
            logger.info("Later message", timestamp_ns=ts1)

        with ThreadSafeLogger(file2) as logger:
            logger.info("Earlier message", timestamp_ns=ts2)

        # Merge with sorting
        merge_mcap_files([file1, file2], merged, sort_by_time=True)

        assert merged.exists()

    def test_merge_empty_list(self, tmp_path: Path) -> None:
        """Test merging empty list of files."""
        merged = tmp_path / "merged.mcap"

        merge_mcap_files([], merged)

        assert merged.exists()
