#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cavpaper.data import SPLITS
from cavpaper.models import build_model
from scripts.evaluate_models import metrics
from scripts.evaluate_rollouts import extra_channels


def load_test_cases():
    cases = []
    for path in sorted((ROOT / "data/processed/cases").glob("naca*.npz")):
        data = {key: value for key, value in np.load(path).items()}
        meta = json.loads(str(data["metadata"]))
        if meta["geometry"] in SPLITS["test"]:
            cases.append((data, meta))
    return cases


def summarize_horizons(model, cases, mean, std, device):
    by_step = {}
    with torch.no_grad():
        for data, meta in cases:
            start = int(np.searchsorted(data["times"], 0.010 - 1e-8))
            state = ((data["fields"][start] - mean) / std).astype(np.float32)
            for step, index in enumerate(range(start, len(data["times"]) - 1), 1):
                inp = np.concatenate([state, extra_channels(data, meta, index)], -1)
                state = model(torch.from_numpy(inp.transpose(2, 0, 1)[None]).to(device)).cpu().numpy()[0].transpose(1, 2, 0)
                by_step.setdefault(step, [[], [], []])
                by_step[step][0].append(state * std + mean)
                by_step[step][1].append(data["fields"][index + 1])
                by_step[step][2].append(data["fluid_mask"][None])
    return {
        step: metrics(np.asarray(values[0]).transpose(0, 3, 1, 2), np.asarray(values[1]).transpose(0, 3, 1, 2), np.asarray(values[2]))
        for step, values in by_step.items()
    }


def main():
    stats = json.loads((ROOT / "data/processed/normalization.json").read_text())
    mean = np.asarray(stats["mean"], np.float32)
    std = np.asarray(stats["std"], np.float32)
    cases = load_test_cases()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    for checkpoint in sorted((ROOT / "training_runs").glob("*/best.pt")):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(payload["model_name"])
        model.load_state_dict(payload["model"])
        model.to(device).eval()
        seed = int(checkpoint.parent.name.split("seed")[-1])
        for step, values in summarize_horizons(model, cases, mean, std, device).items():
            rows.append({"model": payload["model_name"], "seed": seed, "step": step, **values})

    for data, _ in cases:
        start = int(np.searchsorted(data["times"], 0.010 - 1e-8))
        initial = data["fields"][start]
        for step, index in enumerate(range(start, len(data["times"]) - 1), 1):
            values = metrics(initial[None].transpose(0, 3, 1, 2), data["fields"][index + 1][None].transpose(0, 3, 1, 2), data["fluid_mask"][None, None])
            rows.append({"model": "persistence", "seed": -1, "step": step, **values})
    (ROOT / "results/horizon_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
