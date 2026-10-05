#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from cavpaper.case import load_cases


def container_path(path: Path) -> str:
    return "/home/openfoam/" + str(path.resolve().relative_to(REPO_ROOT.resolve()))


def set_patch_type(path: Path, patch_name: str, patch_type: str) -> None:
    text = path.read_text()
    marker = f"    {patch_name}\n    {{"
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"{patch_name} patch missing from {path}")
    end = text.find("    }", start)
    block = text[start:end]
    block = block.replace("type            patch;", f"type            {patch_type};")
    block = block.replace("physicalType    patch;", f"physicalType    {patch_type};")
    path.write_text(text[:start] + block + text[end:])


def run(command: str) -> None:
    subprocess.run([
        "docker", "run", "--rm", "--platform", "linux/amd64", "--entrypoint", "/bin/bash",
        "-v", f"{REPO_ROOT.resolve()}:/home/openfoam",
        "-w", "/home/openfoam",
        "microfluidica/openfoam:2312", "-c",
        f"source /usr/lib/openfoam/openfoam2312/etc/bashrc && {command}",
    ], check=True)


def clean_previous_run(case_dir: Path) -> None:
    for child in case_dir.iterdir():
        if child.is_dir() and child.name != "0":
            try:
                float(child.name)
            except ValueError:
                continue
            shutil.rmtree(child)
    shutil.rmtree(case_dir / "postProcessing", ignore_errors=True)
    shutil.rmtree(case_dir / "VTK", ignore_errors=True)
    for log in case_dir.glob("log.*"):
        log.unlink()


def run_case(case, data_dir: Path, mesh_only: bool) -> None:
    case_dir = data_dir / case.id
    clean_previous_run(case_dir)
    mesh = case_dir / "mesh.msh"
    subprocess.run(["gmsh", "-3", "-format", "msh2", str(case_dir / "mesh.geo"), "-o", str(mesh)], check=True)
    ccase = container_path(case_dir)
    run(f"gmshToFoam -case {ccase} {container_path(mesh)} > {ccase}/log.gmshToFoam")
    boundary = case_dir / "constant" / "polyMesh" / "boundary"
    set_patch_type(boundary, "frontAndBack", "empty")
    set_patch_type(boundary, "airfoil", "wall")
    run(f"checkMesh -case {ccase} > {ccase}/log.checkMesh")
    if not mesh_only:
        print(f"[running] {case.id}", flush=True)
        run(f"potentialFoam -case {ccase} -pName p_rgh -writephi > {ccase}/log.potentialFoam")
        run(f"interPhaseChangeFoam -case {ccase} > {ccase}/log.interPhaseChangeFoam")
        print(f"[complete] {case.id}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--mesh-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    cases = load_cases(args.config)
    if args.jobs == 1:
        for case in cases:
            run_case(case, args.data_dir, args.mesh_only)
        return
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_case, case, args.data_dir, args.mesh_only): case.id for case in cases}
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
