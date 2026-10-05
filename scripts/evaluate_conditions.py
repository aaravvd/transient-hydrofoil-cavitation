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


def test_cases():
    for path in sorted((ROOT / "data/processed/cases").glob("naca*.npz")):
        data = {key: value for key, value in np.load(path).items()}
        meta = json.loads(str(data["metadata"]))
        if meta["geometry"] in SPLITS["test"]:
            yield data, meta


def rollout(model, data, meta, mean, std, device):
    start = int(np.searchsorted(data["times"], 0.010 - 1e-8))
    state = ((data["fields"][start] - mean) / std).astype(np.float32)
    predictions, truths, masks = [], [], []
    with torch.no_grad():
        for index in range(start, len(data["times"]) - 1):
            inp = np.concatenate([state, extra_channels(data, meta, index)], -1)
            state = model(torch.from_numpy(inp.transpose(2, 0, 1)[None]).to(device)).cpu().numpy()[0].transpose(1, 2, 0)
            predictions.append(state * std + mean)
            truths.append(data["fields"][index + 1])
            masks.append(data["fluid_mask"][None])
    return metrics(np.asarray(predictions).transpose(0, 3, 1, 2), np.asarray(truths).transpose(0, 3, 1, 2), np.asarray(masks))


def main():
    stats = json.loads((ROOT / "data/processed/normalization.json").read_text())
    mean = np.asarray(stats["mean"], np.float32)
    std = np.asarray(stats["std"], np.float32)
    cases = list(test_cases())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    for checkpoint in sorted((ROOT / "training_runs").glob("*/best.pt")):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(payload["model_name"])
        model.load_state_dict(payload["model"])
        model.to(device).eval()
        seed = int(checkpoint.parent.name.split("seed")[-1])
        for data, meta in cases:
            rows.append({"model": payload["model_name"], "seed": seed, "angle_deg": meta["angle_deg"], "cavitation_number": meta["cavitation_number"], **rollout(model, data, meta, mean, std, device)})

    for data, meta in cases:
        start = int(np.searchsorted(data["times"], 0.010 - 1e-8))
        count = len(data["times"]) - start - 1
        pred = np.repeat(data["fields"][start][None], count, axis=0)
        truth = data["fields"][start + 1:]
        mask = np.repeat(data["fluid_mask"][None, None], count, axis=0)
        rows.append({"model": "persistence", "seed": -1, "angle_deg": meta["angle_deg"], "cavitation_number": meta["cavitation_number"], **metrics(pred.transpose(0, 3, 1, 2), truth.transpose(0, 3, 1, 2), mask)})
    (ROOT / "results/condition_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
