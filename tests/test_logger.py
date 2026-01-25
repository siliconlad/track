"""Tests for the Logger class."""

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
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
                0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
                0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
                0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,  # IDAT chunk
                0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x05, 0x6D,
                0xC5, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
                0x44, 0xAE, 0x42, 0x60, 0x82,  # IEND chunk
            ]
        )

        with Logger(output_file) as logger:
            logger.log_image("test_image", png_data, format="png")

        assert output_file.exists()

    def test_log_image_array(self, tmp_path: Path) -> None:
        """Test logging an image from numpy array."""
        pytest.importorskip("PIL")
        output_file = tmp_path / "test.mcap"

        # Create a simple RGB image
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # Red channel

        with Logger(output_file) as logger:
            logger.log_image("test_image", image, format="png")

        assert output_file.exists()

    def test_log_pointcloud(self, tmp_path: Path) -> None:
        """Test logging a point cloud with structured array."""
        output_file = tmp_path / "test.mcap"

        # Create structured array with XYZ
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
        points = np.zeros(3, dtype=dtype)
        points["x"] = [0.0, 1.0, 0.0]
        points["y"] = [0.0, 0.0, 1.0]
        points["z"] = [0.0, 0.0, 0.0]

        with Logger(output_file) as logger:
            logger.log_pointcloud("test_pc", points)

        assert output_file.exists()

    def test_log_pointcloud_with_colors(self, tmp_path: Path) -> None:
        """Test logging a point cloud with colors."""
        output_file = tmp_path / "test.mcap"

        # Create structured array with XYZ and RGB
        dtype = np.dtype([
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ])
        points = np.zeros(3, dtype=dtype)
        points["x"] = [0.0, 1.0, 0.0]
        points["y"] = [0.0, 0.0, 1.0]
        points["z"] = [0.0, 0.0, 0.0]
        points["r"] = [255, 0, 0]
        points["g"] = [0, 255, 0]
        points["b"] = [0, 0, 255]

        with Logger(output_file) as logger:
            logger.log_pointcloud("test_pc", points)

        assert output_file.exists()

    def test_log_pointcloud_with_intensity(self, tmp_path: Path) -> None:
        """Test logging a point cloud with intensity."""
        output_file = tmp_path / "test.mcap"

        # Create structured array with XYZ and intensity
        dtype = np.dtype([
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("intensity", "f4"),
        ])
        points = np.zeros(3, dtype=dtype)
        points["x"] = [0.0, 1.0, 0.0]
        points["y"] = [0.0, 0.0, 1.0]
        points["z"] = [0.0, 0.0, 0.0]
        points["intensity"] = [0.5, 0.8, 1.0]

        with Logger(output_file) as logger:
            logger.log_pointcloud("test_pc", points)

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

    def test_invalid_pointcloud_not_structured(self, tmp_path: Path) -> None:
        """Test that non-structured arrays raise errors."""
        output_file = tmp_path / "test.mcap"

        # Regular array, not structured
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

        with Logger(output_file) as logger:
            with pytest.raises(ValueError, match="structured array"):
                logger.log_pointcloud("test_pc", points)

    def test_not_open_error(self, tmp_path: Path) -> None:
        """Test that operations on a closed logger raise errors."""
        output_file = tmp_path / "test.mcap"
        logger = Logger(output_file)

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
        """Test that log level values are correct."""
        assert LogLevel.UNKNOWN == 0
        assert LogLevel.DEBUG == 1
        assert LogLevel.INFO == 2
        assert LogLevel.WARNING == 3
        assert LogLevel.ERROR == 4
        assert LogLevel.FATAL == 5
