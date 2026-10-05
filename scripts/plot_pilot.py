#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
case_path = ROOT / "data" / "processed" / "cases" / "pilot_naca0012_a07_s120.npz"
output = ROOT / "results" / "pilot_fields.png"

data = np.load(case_path)
fields = data["fields"]
x, y = data["x"], data["y"]
frame = fields[-1]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8), constrained_layout=True)
items = [
    (frame[..., 2] / 1000.0, "Pressure", "kPa", "coolwarm"),
    (frame[..., 3], "Vapor fraction", r"$\alpha_v$", "viridis"),
    (np.linalg.norm(frame[..., :2], axis=-1), "Velocity magnitude", "m/s", "magma"),
]
for axis, (values, title, label, cmap) in zip(axes, items):
    image = axis.pcolormesh(x / 0.1, y / 0.1, values, shading="auto", cmap=cmap)
    axis.set_title(title)
    axis.set_xlabel("x/c")
    axis.set_aspect("equal")
    fig.colorbar(image, ax=axis, label=label, shrink=0.8)
axes[0].set_ylabel("y/c")
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=220)
print(output)
print(f"alpha max={frame[..., 3].max():.6f}, alpha>0.1 cells={(frame[..., 3] > 0.1).sum()}")
