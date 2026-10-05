#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cavpaper.data import CavitationDataset
from cavpaper.models import build_model


def metrics(pred, truth, mask):
    by_channel = {}
    names = ("u", "v", "p", "alpha_v")
    for channel, name in enumerate(names):
        keep = mask[:, 0].astype(bool)
        delta = pred[:, channel][keep] - truth[:, channel][keep]
        by_channel[f"{name}_rmse"] = float(np.sqrt(np.mean(delta**2)))
        by_channel[f"{name}_mae"] = float(np.mean(np.abs(delta)))
    pbin = (pred[:, 3] > 0.1) & mask[:, 0].astype(bool)
    tbin = (truth[:, 3] > 0.1) & mask[:, 0].astype(bool)
    intersection = (pbin & tbin).sum()
    by_channel["vapor_iou"] = float(intersection / max((pbin | tbin).sum(), 1))
    by_channel["vapor_dice"] = float(2*intersection / max(pbin.sum()+tbin.sum(), 1))
    by_channel["cavity_volume_mae"] = float(np.mean(np.abs(pred[:, 3].clip(0, 1).sum((1,2))-truth[:, 3].sum((1,2)))))
    return by_channel


def main():
    stats = json.loads((ROOT/"data/processed/normalization.json").read_text())
    mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
    dataset = CavitationDataset(ROOT/"data/processed/cases", "test", mean, std)
    loader = DataLoader(dataset, batch_size=8)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    for checkpoint in sorted((ROOT/"training_runs").glob("*/best.pt")):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(payload["model_name"]); model.load_state_dict(payload["model"]); model.to(device).eval()
        predictions, truths, masks, elapsed = [], [], [], 0.0
        with torch.no_grad():
            for inp, target, mask in loader:
                inp = inp.to(device)
                if device.type == "mps": torch.mps.synchronize()
                start = time.perf_counter(); output = model(inp)
                if device.type == "mps": torch.mps.synchronize()
                elapsed += time.perf_counter()-start
                predictions.append(output.cpu().numpy().transpose(0,2,3,1)*std+mean)
                truths.append(target.numpy().transpose(0,2,3,1)*std+mean); masks.append(mask.numpy())
        pred, truth, mask = np.concatenate(predictions), np.concatenate(truths), np.concatenate(masks)
        row = {"model": payload["model_name"], "seed": int(checkpoint.parent.name.split("seed")[-1]), "parameters": sum(p.numel() for p in model.parameters()), "inference_ms_per_frame": 1000*elapsed/len(dataset)}
        row.update(metrics(pred.transpose(0,3,1,2), truth.transpose(0,3,1,2), mask)); rows.append(row)
    # Persistence is a necessary transient baseline.
    current, truth, masks = [], [], []
    for inp, target, mask in loader:
        current.append(inp[:, :4].numpy().transpose(0,2,3,1)*std+mean)
        truth.append(target.numpy().transpose(0,2,3,1)*std+mean); masks.append(mask.numpy())
    row = {"model":"persistence", "seed":-1, "parameters":0, "inference_ms_per_frame":0.0}
    row.update(metrics(np.concatenate(current).transpose(0,3,1,2), np.concatenate(truth).transpose(0,3,1,2), np.concatenate(masks))); rows.append(row)
    output = ROOT/"results"; output.mkdir(exist_ok=True)
    (output/"test_metrics.json").write_text(json.dumps(rows, indent=2)+"\n")
    with (output/"test_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()
