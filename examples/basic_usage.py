#!/usr/bin/env python3
"""Basic usage example for the track library.

This example demonstrates how to use the Logger to track:
- Text logs with different severity levels
- Images (both raw bytes and numpy arrays)
- Point clouds with flexible field definitions

The output MCAP file can be visualized in Foxglove Studio.
"""

import numpy as np

from track import Logger


def generate_sample_pointcloud(n_points: int = 1000) -> np.ndarray:
    """Generate a sample point cloud (sphere with random colors)."""
    # Define structured dtype with XYZ and RGB
    dtype = np.dtype([
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    points = np.zeros(n_points, dtype=dtype)

    # Generate points on a sphere
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = np.random.uniform(0.8, 1.2, n_points)

    points["x"] = r * np.sin(phi) * np.cos(theta)
    points["y"] = r * np.sin(phi) * np.sin(theta)
    points["z"] = r * np.cos(phi)

    # Generate random colors
    points["r"] = np.random.randint(0, 255, n_points)
    points["g"] = np.random.randint(0, 255, n_points)
    points["b"] = np.random.randint(0, 255, n_points)

    return points


def generate_sample_image(width: int = 256, height: int = 256) -> np.ndarray:
    """Generate a sample gradient image."""
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)

    # Create RGB image with gradients
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = xv  # Red channel: horizontal gradient
    image[:, :, 1] = yv  # Green channel: vertical gradient
    image[:, :, 2] = 128  # Blue channel: constant

    return image


def main():
    """Run the example."""
    output_file = "example_experiment.mcap"

    with Logger(output_file, name="example") as logger:
        # Add experiment metadata
        logger.add_metadata(
            "experiment",
            {
                "name": "example_experiment",
                "model": "resnet50",
                "learning_rate": "0.001",
                "batch_size": "32",
            },
        )

        # Log some messages
        logger.info("Starting experiment")
        logger.debug("Debug: Loading configuration")
        logger.info("Training started with batch_size=32")

        # Simulate training loop
        for epoch in range(3):
            logger.info(f"Epoch {epoch + 1}/3 started")

            # Generate and log a sample point cloud
            points = generate_sample_pointcloud(500)
            logger.log_pointcloud(
                topic="training/features",
                points=points,
                frame_id="model_space",
            )

            # Generate and log a sample image
            image = generate_sample_image()
            logger.log_image(
                topic="training/activations",
                image=image,
                format="png",
                frame_id="image",
            )

            # Simulate some training metrics
            loss = 1.0 / (epoch + 1) + np.random.uniform(-0.1, 0.1)
            accuracy = 0.5 + epoch * 0.15 + np.random.uniform(-0.05, 0.05)

            logger.info(f"Epoch {epoch + 1} complete: loss={loss:.4f}, accuracy={accuracy:.4f}")

            if loss > 0.5:
                logger.warning(f"Loss is still high: {loss:.4f}")

        logger.info("Training complete!")

        # Log an error example
        logger.error("Example error message for demonstration")

    print(f"Experiment logged to: {output_file}")
    print("Open this file in Foxglove Studio to visualize the data.")


if __name__ == "__main__":
    main()
