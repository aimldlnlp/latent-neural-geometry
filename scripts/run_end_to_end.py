#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_manifold.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the neural manifold recovery pipeline.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "default.yaml"), help="Path to a YAML config file.")
    parser.add_argument("--output", type=str, default=None, help="Optional explicit output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(config_path=args.config, output_dir=args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
