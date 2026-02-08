#!/usr/bin/env python3
"""Example of queue-based logging with multiple processes.

When `use_process=True`, Logger uses a dedicated writer process and
multiprocessing queue so producers can log safely from many processes.
"""

import multiprocessing as mp
import time

import numpy as np

from track import Logger


def generate_pointcloud(n_points: int = 50) -> np.ndarray:
    """Generate a sample point cloud with colors."""
    dtype = np.dtype([
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    points = np.zeros(n_points, dtype=dtype)
    points["x"] = np.random.randn(n_points)
    points["y"] = np.random.randn(n_points)
    points["z"] = np.random.randn(n_points)
    points["r"] = np.random.randint(0, 255, n_points)
    points["g"] = np.random.randint(0, 255, n_points)
    points["b"] = np.random.randint(0, 255, n_points)
    return points


def worker(logger: Logger, worker_id: int, num_iterations: int) -> None:
    """Worker function that runs in a separate process."""
    for i in range(num_iterations):
        logger.info(f"Process {worker_id}: iteration {i}")

        if i % 10 == 0:
            logger.debug(f"Process {worker_id}: checkpoint at {i}")

        # Simulate computation
        time.sleep(0.02)

        # Log some data periodically
        if i % 20 == 0:
            points = generate_pointcloud(50)
            logger.log_pointcloud(
                f"process_{worker_id}/data",
                points,
                frame_id="world",
            )

    logger.info(f"Process {worker_id}: finished")


def main():
    """Run the multi-process logging example."""
    output_file = "multiprocess_experiment.mcap"
    num_processes = 4
    iterations_per_process = 30

    print(f"Starting {num_processes} processes, {iterations_per_process} iterations each")

    with Logger(output_file, name="multiprocess", use_process=True) as logger:
        # Add metadata
        logger.add_metadata(
            "experiment",
            {
                "type": "multiprocess",
                "num_processes": str(num_processes),
                "iterations": str(iterations_per_process),
            },
        )

        logger.info("Starting multi-process experiment")

        # Spawn worker processes
        processes = [
            mp.Process(target=worker, args=(logger, i, iterations_per_process))
            for i in range(num_processes)
        ]

        start_time = time.time()

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        elapsed = time.time() - start_time
        logger.info(f"All processes completed in {elapsed:.2f}s")

    print(f"Logged to: {output_file}")
    print(f"Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    # Required for Windows compatibility
    mp.set_start_method("spawn", force=True)
    main()
