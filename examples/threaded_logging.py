#!/usr/bin/env python3
"""Example of async logging with multiple threads.

This example demonstrates how to use AsyncLogger with multiple threads.
The AsyncLogger uses a background writer thread, so log calls return
immediately without blocking.
"""

import threading
import time

import numpy as np

from track import AsyncLogger


def worker(logger: AsyncLogger, worker_id: int, num_iterations: int) -> None:
    """Worker function that logs messages from a thread."""
    for i in range(num_iterations):
        # Log calls return immediately (non-blocking)
        logger.info(f"Worker {worker_id}: iteration {i}")

        if i % 10 == 0:
            logger.debug(f"Worker {worker_id}: debug info at iteration {i}")

        # Simulate some work
        time.sleep(0.01)

        # Occasionally log a point cloud
        if i % 25 == 0:
            points = np.random.randn(100, 3).astype(np.float32)
            logger.log_pointcloud(
                f"worker_{worker_id}/features",
                points,
                frame_id="local",
            )

    logger.info(f"Worker {worker_id}: completed")


def main():
    """Run the threaded logging example."""
    output_file = "threaded_experiment.mcap"
    num_threads = 4
    iterations_per_thread = 50

    print(f"Starting {num_threads} threads, {iterations_per_thread} iterations each")

    # AsyncLogger uses a background writer thread by default
    with AsyncLogger(output_file, name="threaded") as logger:
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

        # Give background writer time to flush
        time.sleep(0.5)

    print(f"Logged to: {output_file}")
    print(f"Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
