#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cavpaper.case import load_cases, write_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()
    for case in load_cases(args.config):
        case_dir = args.output_dir / case.id
        write_case(case, case_dir)
        print(f"[generated] {case.id}: U={case.speed:.4g} m/s, p_out={case.outlet_pressure:.4g} Pa")


if __name__ == "__main__":
    main()

