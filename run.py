#!/usr/bin/env python3
"""Entry point for the Scorecards for Extremes pipeline.

The pipeline code lives in ``src/``. This thin wrapper puts ``src/`` on the
import path so the historical invocation still works unchanged:

    python run.py <config_file.yaml>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from run import main  # noqa: E402  (import after sys.path is set up)

if __name__ == "__main__":
    main()
