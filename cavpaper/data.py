from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SPLITS = {"train": ("NACA0012", "NACA2412", "NACA4412"), "val": ("NACA0015",), "test": ("NACA2415",)}


def load_records(case_dir: Path, split: str, warmup=0.010):
    records = []
    for path in sorted(case_dir.glob("*.npz")):
        if path.name.startswith("pilot_"):
            continue
        data = np.load(path)
        meta = json.loads(str(data["metadata"]))
        if meta["geometry"] not in SPLITS[split]:
            continue
        fields, times = data["fields"], data["times"]
        for i in range(len(times)-1):
            if times[i] >= warmup - 1e-8:
                records.append((path, i))
    return records


def compute_stats(case_dir: Path):
    values = []
    records = load_records(case_dir, "train")
    for path in sorted({path for path, _ in records}):
        data = np.load(path)
        mask = data["fluid_mask"].astype(bool)
        indices = [i for record_path, i in records if record_path == path]
        values.extend(data["fields"][i+1][mask] for i in indices)
    stacked = np.concatenate(values)
    return stacked.mean(0).astype(np.float32), np.maximum(stacked.std(0), 1e-6).astype(np.float32)


class CavitationDataset(Dataset):
    def __init__(self, case_dir: Path, split: str, mean, std):
        self.records = load_records(case_dir, split)
        self.mean, self.std = np.asarray(mean), np.asarray(std)
        self.cache = {path: {key: value for key, value in np.load(path).items()} for path, _ in self.records}

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        path, i = self.records[index]
        data = self.cache[path]
        meta = json.loads(str(data["metadata"]))
        current = (data["fields"][i] - self.mean) / self.std
        target = (data["fields"][i+1] - self.mean) / self.std
        xg, yg = np.meshgrid(data["x"] / meta["chord_m"], data["y"] / meta["chord_m"])
        h, w = xg.shape
        extras = np.stack([
            xg, yg, data["fluid_mask"],
            np.full((h, w), meta["angle_deg"] / 10),
            np.full((h, w), meta["cavitation_number"] / 2),
            np.full((h, w), float(data["times"][i]) / 0.03),
        ], -1)
        inp = np.concatenate([current, extras], -1).transpose(2, 0, 1).astype(np.float32)
        return torch.from_numpy(inp), torch.from_numpy(target.transpose(2, 0, 1).astype(np.float32)), torch.from_numpy(data["fluid_mask"][None].astype(np.float32))


class CavitationPhysicsDataset(CavitationDataset):
    """Adds current eddy viscosity and dimensional grid/time spacing for PINN residuals."""

    def __getitem__(self, index):
        inp, target, mask = super().__getitem__(index)
        path, i = self.records[index]
        data = self.cache[path]
        dx = float(data["x"][1] - data["x"][0])
        dy = float(data["y"][1] - data["y"][0])
        dt = float(data["times"][i + 1] - data["times"][i])
        nut = torch.from_numpy(data["nut"][i][None].astype(np.float32))
        spacing = torch.tensor([dx, dy, dt], dtype=torch.float32)
        return inp, target, mask, nut, spacing
