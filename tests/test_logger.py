"""Tests for the Logger class."""

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from track import Logger, LogLevel


class TestLogger:
    """Tests for the Logger class."""

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test that Logger works as a context manager."""
        output_file = tmp_path / "test.mcap"

        with Logger(output_file) as logger:
            logger.info("Test message")

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_log_levels(self, tmp_path: Path) -> None:
        """Test all log levels."""
        output_file = tmp_path / "test.mcap"

        with Logger(output_file, name="test") as logger:
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.fatal("Fatal message")

        assert output_file.exists()

    def test_log_image_bytes(self, tmp_path: Path) -> None:
        """Test logging an image from bytes."""
        output_file = tmp_path / "test.mcap"

        # Create a minimal PNG (1x1 red pixel)
        png_data = bytes(
            [
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,  # PNG signature
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,  # IHDR chunk
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,
                0x08,
                0x02,
                0x00,
                0x00,
                0x00,
                0x90,
                0x77,
                0x53,
                0xDE,
                0x00,
                0x00,
                0x00,
                0x0C,
                0x49,
                0x44,
                0x41,
                0x54,  # IDAT chunk
                0x08,
                0xD7,
                0x63,
                0xF8,
                0xCF,
                0xC0,
                0x00,
                0x00,
                0x00,
                0x03,
                0x00,
                0x01,
                0x00,
                0x05,
                0x6D,
                0xC5,
                0x00,
                0x00,
                0x00,
                0x00,
                0x49,
                0x45,
                0x4E,
                0x44,  # IEND chunk
                0xAE,
                0x42,
                0x60,
                0x82,
            ]
        )

        with Logger(output_file) as logger:
            logger.log_image("test_image", png_data, format="png")

        assert output_file.exists()

    def test_log_pointcloud(self, tmp_path: Path) -> None:
        """Test logging a point cloud."""
        output_file = tmp_path / "test.mcap"

        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        with Logger(output_file) as logger:
            logger.log_pointcloud("test_pc", points)

        assert output_file.exists()

    def test_log_pointcloud_with_colors(self, tmp_path: Path) -> None:
        """Test logging a point cloud with colors."""
        output_file = tmp_path / "test.mcap"

        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

        with Logger(output_file) as logger:
            logger.log_pointcloud("test_pc", points, colors=colors)

        assert output_file.exists()

    def test_log_pointcloud_with_intensities(self, tmp_path: Path) -> None:
        """Test logging a point cloud with intensities."""
        output_file = tmp_path / "test.mcap"

        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        intensities = np.array([0.5, 0.8, 1.0], dtype=np.float32)

        with Logger(output_file) as logger:
            logger.log_pointcloud("test_pc", points, intensities=intensities)

        assert output_file.exists()

    def test_add_metadata(self, tmp_path: Path) -> None:
        """Test adding metadata."""
        output_file = tmp_path / "test.mcap"

        with Logger(output_file) as logger:
            logger.add_metadata("experiment", {"name": "test", "version": "1.0"})

        assert output_file.exists()

    def test_add_attachment(self, tmp_path: Path) -> None:
        """Test adding an attachment."""
        output_file = tmp_path / "test.mcap"

        with Logger(output_file) as logger:
            logger.add_attachment("config.json", b'{"key": "value"}', media_type="application/json")

        assert output_file.exists()

    def test_file_object_output(self, tmp_path: Path) -> None:
        """Test writing to a file object."""
        output_file = tmp_path / "test.mcap"

        with open(output_file, "wb") as f:
            with Logger(f) as logger:
                logger.info("Test message")

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_invalid_pointcloud_shape(self, tmp_path: Path) -> None:
        """Test that invalid point cloud shapes raise errors."""
        output_file = tmp_path / "test.mcap"

        points = np.array([1.0, 2.0, 3.0])  # 1D array, should be Nx3

        with Logger(output_file) as logger:
            with pytest.raises(ValueError, match="must be Nx3"):
                logger.log_pointcloud("test_pc", points)

    def test_not_open_error(self) -> None:
        """Test that operations on a closed logger raise errors."""
        logger = Logger(io.BytesIO())

        with pytest.raises(RuntimeError, match="not open"):
            logger.info("Test")

    def test_compression_options(self, tmp_path: Path) -> None:
        """Test different compression options."""
        for compression in ["zstd", "lz4", "none"]:
            output_file = tmp_path / f"test_{compression}.mcap"

            with Logger(output_file, compression=compression) as logger:
                logger.info("Test message")

            assert output_file.exists()


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_level_values(self) -> None:
        """Test that log level values match Foxglove spec."""
        assert LogLevel.UNKNOWN == 0
        assert LogLevel.DEBUG == 1
        assert LogLevel.INFO == 2
        assert LogLevel.WARNING == 3
        assert LogLevel.ERROR == 4
        assert LogLevel.FATAL == 5
