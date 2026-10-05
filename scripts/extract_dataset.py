#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from cavpaper.foamio import numeric_times, read_internal_field
from cavpaper.case import naca4_coordinates


def container_path(path: Path) -> str:
    return "/home/openfoam/" + str(path.resolve().relative_to(REPO_ROOT.resolve()))


def write_centres(case_dir: Path) -> None:
    times = numeric_times(case_dir)
    if times and (times[-1] / "C").exists():
        return
    subprocess.run([
        "docker", "run", "--rm", "--platform", "linux/amd64", "--entrypoint", "/bin/bash",
        "-v", f"{REPO_ROOT.resolve()}:/home/openfoam", "-w", "/home/openfoam",
        "microfluidica/openfoam:2312", "-c",
        "source /usr/lib/openfoam/openfoam2312/etc/bashrc && "
        f"postProcess -case {container_path(case_dir)} -func writeCellCentres -latestTime >/dev/null",
    ], check=True)


def interpolate(points: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    linear = griddata(points, values, query, method="linear")
    missing = np.isnan(linear)
    if np.any(missing):
        linear[missing] = griddata(points, values, query[missing], method="nearest")
    return linear


def inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        crossing = ((y1 > y) != (y0 > y)) & (
            x < (x0 - x1) * (y - y1) / (y0 - y1 + 1e-30) + x1
        )
        inside ^= crossing
        x0, y0 = x1, y1
    return inside


def extract_case(case_dir: Path, nx: int, ny: int) -> dict[str, np.ndarray]:
    metadata = json.loads((case_dir / "case.json").read_text())
    chord = metadata["chord_m"]
    times = numeric_times(case_dir)
    if not times:
        raise RuntimeError(f"No completed times in {case_dir}")
    write_centres(case_dir)
    centres = read_internal_field(times[-1] / "C")[:, :2]
    x = np.linspace(-0.5 * chord, 1.5 * chord, nx)
    y = np.linspace(-0.5 * chord, 0.5 * chord, ny)
    xx, yy = np.meshgrid(x, y)
    query = np.column_stack([xx.ravel(), yy.ravel()])
    foil_x, foil_y = naca4_coordinates(metadata["geometry"], chord=chord)
    solid = inside_polygon(query, np.column_stack([foil_x, foil_y])).reshape(ny, nx)

    frames = []
    nut_frames = []
    for time_dir in times:
        u = read_internal_field(time_dir / "U")[:, :2]
        p = read_internal_field(time_dir / "p")
        alpha = 1.0 - read_internal_field(time_dir / "alpha.water")
        channels = np.column_stack([u, p, alpha])
        frame = interpolate(centres, channels, query).reshape(ny, nx, 4)
        frames.append(frame)
        nut = read_internal_field(time_dir / "nut")
        nut_frames.append(interpolate(centres, nut, query).reshape(ny, nx))
    return {
        "fields": np.asarray(frames, dtype=np.float32),
        "nut": np.asarray(nut_frames, dtype=np.float32),
        "times": np.asarray([float(path.name) for path in times], dtype=np.float32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "metadata": np.asarray(json.dumps(metadata)),
        "fluid_mask": (~solid).astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "cases")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = [args.raw_dir / name for name in args.case] if args.case else sorted(args.raw_dir.glob("*"))
    for case_dir in case_dirs:
        if not (case_dir / "case.json").exists():
            continue
        payload = extract_case(case_dir, args.nx, args.ny)
        output = args.output_dir / f"{case_dir.name}.npz"
        np.savez_compressed(output, **payload)
        print(f"[extracted] {case_dir.name}: {payload['fields'].shape} -> {output}")


if __name__ == "__main__":
    main()
