from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def read_internal_field(path: Path) -> np.ndarray:
    text = path.read_text()
    uniform = re.search(r"internalField\s+uniform\s+(\([^;]+\)|[^;\s]+)\s*;", text)
    if uniform:
        token = uniform.group(1).strip()
        if token.startswith("("):
            return np.asarray([[float(value) for value in token[1:-1].split()]])
        return np.asarray([float(token)])

    match = re.search(
        r"internalField\s+nonuniform\s+List<(scalar|vector)>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"Cannot parse internalField in {path}")
    kind, count, body = match.groups()
    if kind == "scalar":
        values = np.fromstring(body, sep=" ")
    else:
        rows = re.findall(r"\(([^()]*)\)", body)
        values = np.asarray([[float(value) for value in row.split()] for row in rows])
    if len(values) != int(count):
        raise ValueError(f"Expected {count} values in {path}, found {len(values)}")
    return values


def numeric_times(case_dir: Path) -> list[Path]:
    times: list[tuple[float, Path]] = []
    for path in case_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        if value > 0:
            times.append((value, path))
    return [path for _, path in sorted(times)]
