#!/usr/bin/env python3
"""Example of distributed logging with per-process files.

This example demonstrates how to use ProcessLocalLogger for distributed
training scenarios where each process writes to its own file.

This approach has several advantages:
- Zero synchronization overhead between processes
- Each process can write at full speed
- Files can be merged after training completes
- Works across machines in a cluster

For real distributed training, set RANK and WORLD_SIZE environment variables,
or use the rank/world_size arguments to ProcessLocalLogger.
"""

import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from track import ProcessLocalLogger, merge_mcap_files


def simulate_distributed_worker(rank: int, world_size: int, output_dir: str) -> None:
    """Simulate a distributed training worker.

    In real distributed training, this would run on different machines/GPUs.
    """
    # Set environment variables (in real training, these would be set by the launcher)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Create process-local logger
    factory = ProcessLocalLogger(
        output_dir=output_dir,
        base_name="distributed_experiment",
    )

    logger = factory.get_logger()

    try:
        logger.info(f"Rank {rank}/{world_size}: Starting training")

        # Simulate training epochs
        for epoch in range(3):
            logger.info(f"Rank {rank}: Epoch {epoch + 1}/3 started")

            # Simulate batch processing
            for batch in range(10):
                # Log training metrics
                loss = 1.0 / (epoch + 1) + np.random.uniform(-0.1, 0.1)
                logger.debug(f"Rank {rank}: epoch={epoch}, batch={batch}, loss={loss:.4f}")

                # Simulate computation
                time.sleep(0.01)

            # Log epoch summary
            epoch_loss = 1.0 / (epoch + 1)
            logger.info(f"Rank {rank}: Epoch {epoch + 1} complete, loss={epoch_loss:.4f}")

            # Log some data
            features = np.random.randn(100, 3).astype(np.float32)
            logger.log_pointcloud(
                f"rank_{rank}/embeddings",
                features,
                frame_id="feature_space",
            )

        logger.info(f"Rank {rank}: Training complete")

    finally:
        factory.close()


def main():
    """Run the distributed logging example."""
    output_dir = "distributed_logs"
    world_size = 4

    print(f"Simulating distributed training with {world_size} workers")

    # Clean up output directory
    output_path = Path(output_dir)
    if output_path.exists():
        for f in output_path.glob("*.mcap"):
            f.unlink()
    output_path.mkdir(exist_ok=True)

    # Spawn workers (simulating distributed processes)
    processes = [
        mp.Process(target=simulate_distributed_worker, args=(rank, world_size, output_dir))
        for rank in range(world_size)
    ]

    start_time = time.time()

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f}s")

    # List output files
    output_files = list(output_path.glob("*.mcap"))
    print(f"\nGenerated files:")
    for f in sorted(output_files):
        print(f"  - {f} ({f.stat().st_size / 1024:.1f} KB)")

    # Merge all files
    merged_path = output_path / "combined.mcap"
    print(f"\nMerging files into: {merged_path}")
    merge_mcap_files(output_files, merged_path, sort_by_time=True)
    print(f"Merged file size: {merged_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
