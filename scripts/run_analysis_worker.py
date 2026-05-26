#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.task_queue import run_worker_loop  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Redis-backed analysis worker.")
    parser.add_argument("--max-jobs", type=int, default=None, help="Stop after processing N jobs.")
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval used when the queue is empty.",
    )
    args = parser.parse_args()
    processed = run_worker_loop(
        max_jobs=args.max_jobs,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    print(f"processed_jobs={processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
