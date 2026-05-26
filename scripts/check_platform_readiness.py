#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.runtime_guard import evaluate_runtime_readiness  # noqa: E402


def evaluate_platform_readiness() -> dict[str, object]:
    return evaluate_runtime_readiness()


def main() -> int:
    print(json.dumps(evaluate_platform_readiness(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
