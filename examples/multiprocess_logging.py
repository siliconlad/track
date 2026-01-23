#!/usr/bin/env python3
"""Example of async logging with multiple processes.

This example demonstrates how to use AsyncLogger with multiple processes.
When use_process=True, the logger uses a dedicated writer process with
a multiprocessing.Queue, making it safe for multi-process applications.
"""

import multiprocessing as mp
import time

import numpy as np

from track import AsyncLogger


def worker(logger: AsyncLogger, worker_id: int, num_iterations: int) -> None:
    """Worker function that runs in a separate process."""
    for i in range(num_iterations):
        # Log calls return immediately (non-blocking)
        logger.info(f"Process {worker_id}: iteration {i}")

        if i % 10 == 0:
            logger.debug(f"Process {worker_id}: checkpoint at {i}")

        # Simulate computation
        time.sleep(0.02)

        # Log some data periodically
        if i % 20 == 0:
            points = np.random.randn(50, 3).astype(np.float32)
            colors = np.random.randint(0, 255, (50, 3), dtype=np.uint8)
            logger.log_pointcloud(
                f"process_{worker_id}/data",
                points,
                colors=colors,
                frame_id="world",
            )

    logger.info(f"Process {worker_id}: finished")


def main():
    """Run the multi-process logging example."""
    output_file = "multiprocess_experiment.mcap"
    num_processes = 4
    iterations_per_process = 30

    print(f"Starting {num_processes} processes, {iterations_per_process} iterations each")

    # Use use_process=True for multi-process safety
    with AsyncLogger(output_file, name="multiprocess", use_process=True) as logger:
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

        # Give background writer time to flush
        time.sleep(0.5)

    print(f"Logged to: {output_file}")
    print(f"Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    # Required for Windows compatibility
    mp.set_start_method("spawn", force=True)
    main()
