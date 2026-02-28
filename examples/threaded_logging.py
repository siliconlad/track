#!/usr/bin/env python3
"""Example of thread-safe logging with multiple threads.

This example demonstrates sharing a single Logger across worker threads.
Logger serializes writes internally with a lock.
"""

import threading
import time

import numpy as np

from track import Logger


def generate_pointcloud(n_points: int = 100) -> np.ndarray:
    """Generate a sample point cloud."""
    dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
    points = np.zeros(n_points, dtype=dtype)
    points["x"] = np.random.randn(n_points)
    points["y"] = np.random.randn(n_points)
    points["z"] = np.random.randn(n_points)
    return points


def worker(logger: Logger, worker_id: int, num_iterations: int) -> None:
    """Worker function that logs messages from a thread."""
    for i in range(num_iterations):
        logger.info(f"Worker {worker_id}: iteration {i}")

        if i % 10 == 0:
            logger.debug(f"Worker {worker_id}: debug info at iteration {i}")

        # Simulate some work
        time.sleep(0.01)

        # Occasionally log a point cloud
        if i % 25 == 0:
            points = generate_pointcloud(100)
            logger.log_pointcloud(
                f"worker_{worker_id}/features",
                points,
                frame_id="local",
            )

    logger.info(f"Worker {worker_id}: completed")


def main():
    """Run the threaded logging example."""
    output_dir = "logs"
    num_threads = 4
    iterations_per_thread = 50

    print(f"Starting {num_threads} threads, {iterations_per_thread} iterations each")

    with Logger("threaded", output_dir=output_dir) as logger:
        output_path = logger.output_path
        # Add experiment metadata
        logger.add_metadata(
            "experiment",
            {
                "type": "threaded",
                "num_threads": str(num_threads),
                "iterations": str(iterations_per_thread),
            },
        )

        logger.info("Starting threaded experiment")

        # Create and start worker threads
        threads = [
            threading.Thread(target=worker, args=(logger, i, iterations_per_thread))
            for i in range(num_threads)
        ]

        start_time = time.time()

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        elapsed = time.time() - start_time
        logger.info(f"All workers completed in {elapsed:.2f}s")

    print(f"Logged to: {output_path}")
    print(f"Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
