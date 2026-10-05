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


def extra_channels(data, meta, index):
    xg, yg = np.meshgrid(data["x"] / meta["chord_m"], data["y"] / meta["chord_m"])
    h, w = xg.shape
    return np.stack([xg, yg, data["fluid_mask"], np.full((h,w), meta["angle_deg"]/10), np.full((h,w), meta["cavitation_number"]/2), np.full((h,w), float(data["times"][index])/0.03)], -1).astype(np.float32)


def main():
    stats = json.loads((ROOT/"data/processed/normalization.json").read_text())
    mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
    cases = []
    for path in sorted((ROOT/"data/processed/cases").glob("naca*.npz")):
        data = {key: value for key, value in np.load(path).items()}
        meta = json.loads(str(data["metadata"]))
        if meta["geometry"] in SPLITS["test"]: cases.append((data, meta))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    for checkpoint in sorted((ROOT/"training_runs").glob("*/best.pt")):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(payload["model_name"]); model.load_state_dict(payload["model"]); model.to(device).eval()
        predictions, truths, masks = [], [], []
        with torch.no_grad():
            for data, meta in cases:
                start = int(np.searchsorted(data["times"], 0.010-1e-8))
                state = ((data["fields"][start]-mean)/std).astype(np.float32)
                for i in range(start, len(data["times"])-1):
                    inp = np.concatenate([state, extra_channels(data, meta, i)], -1).transpose(2,0,1)[None]
                    state = model(torch.from_numpy(inp).to(device)).cpu().numpy()[0].transpose(1,2,0)
                    predictions.append(state*std+mean); truths.append(data["fields"][i+1]); masks.append(data["fluid_mask"][None])
        pred, truth, mask = np.asarray(predictions), np.asarray(truths), np.asarray(masks)
        row = {"model": payload["model_name"], "seed": int(checkpoint.parent.name.split("seed")[-1])}
        row.update(metrics(pred.transpose(0,3,1,2), truth.transpose(0,3,1,2), mask)); rows.append(row)
    predictions, truths, masks = [], [], []
    for data, _ in cases:
        start = int(np.searchsorted(data["times"], 0.010-1e-8)); state = data["fields"][start]
        for i in range(start, len(data["times"])-1):
            predictions.append(state); truths.append(data["fields"][i+1]); masks.append(data["fluid_mask"][None])
    row = {"model":"persistence", "seed":-1}; row.update(metrics(np.asarray(predictions).transpose(0,3,1,2), np.asarray(truths).transpose(0,3,1,2), np.asarray(masks))); rows.append(row)
    (ROOT/"results/rollout_metrics.json").write_text(json.dumps(rows, indent=2)+"\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()
