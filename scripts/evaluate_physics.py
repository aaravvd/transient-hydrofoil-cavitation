#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cavpaper.data import CavitationPhysicsDataset
from cavpaper.models import build_model
from scripts.train_models import multiphase_pinn_residuals


def main():
    stats = json.loads((ROOT / "data/processed/normalization.json").read_text())
    mean = np.asarray(stats["mean"], np.float32)
    std = np.asarray(stats["std"], np.float32)
    dataset = CavitationPhysicsDataset(ROOT / "data/processed/cases", "test", mean, std)
    loader = DataLoader(dataset, batch_size=8)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)
    rows = []
    for checkpoint in sorted((ROOT / "training_runs").glob("*/best.pt")):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(payload["model_name"])
        model.load_state_dict(payload["model"])
        model.to(device).eval()
        totals = {key: 0.0 for key in ("continuity", "x_momentum", "y_momentum", "vapor_transport")}
        with torch.no_grad():
            for inp, _, mask, nut, spacing in loader:
                inp, mask = inp.to(device), mask.to(device)
                residuals = multiphase_pinn_residuals(
                    model(inp), inp, mask, nut.to(device), spacing.to(device), mean_t, std_t
                )
                for key, value in residuals.items():
                    totals[key] += value.item() * len(inp)
        record = {
            "model": payload["model_name"],
            "seed": int(checkpoint.parent.name.split("seed")[-1]),
        }
        record.update({key: value / len(dataset) for key, value in totals.items()})
        record["physics_residual"] = sum(record[key] for key in totals)
        rows.append(record)
    output = ROOT / "results/physics_residuals.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
