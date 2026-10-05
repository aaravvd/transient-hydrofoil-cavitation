#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cavpaper.data import CavitationDataset, CavitationPhysicsDataset, compute_stats
from cavpaper.models import build_model


RHO_L, RHO_V = 997.0, 0.023
NU_L, NU_V = 1.0e-6, 4.3e-4
P_V = 2300.0
NUCLEI_DENSITY, NUCLEI_DIAMETER = 1.6e13, 2.0e-6


def masked_mse(pred, target, mask):
    weights = torch.tensor([1, 1, 1, 5], device=pred.device)[None, :, None, None]
    return (((pred-target)**2 * weights * mask).sum() / (mask.sum() * weights.sum()))


def ddx(field, dx):
    return (field[:, 1:-1, 2:] - field[:, 1:-1, :-2]) / (2 * dx[:, None, None])


def ddy(field, dy):
    return (field[:, 2:, 1:-1] - field[:, :-2, 1:-1]) / (2 * dy[:, None, None])


def laplacian(field, dx, dy):
    center = field[:, 1:-1, 1:-1]
    dxx = (field[:, 1:-1, 2:] - 2 * center + field[:, 1:-1, :-2]) / dx[:, None, None] ** 2
    dyy = (field[:, 2:, 1:-1] - 2 * center + field[:, :-2, 1:-1]) / dy[:, None, None] ** 2
    return dxx + dyy


def schnerr_sauer_alpha_source(alpha_v, pressure):
    alpha_v = alpha_v.clamp(1e-6, 1 - 1e-6)
    alpha_nuc = (np.pi / 6) * NUCLEI_DENSITY * NUCLEI_DIAMETER**3
    alpha_nuc = alpha_nuc / (1 + alpha_nuc)
    radius = ((3 / (4 * np.pi * NUCLEI_DENSITY)) * (alpha_v + alpha_nuc) / (1 - alpha_v)).clamp_min(1e-18).pow(1 / 3)
    pressure_delta = pressure - P_V
    collapse = 3 * alpha_v / radius * torch.sqrt((2 / 3) * pressure_delta.clamp_min(0) / RHO_L)
    vaporize = 3 * (alpha_nuc + alpha_v) * (1 - alpha_v) / radius * torch.sqrt((2 / 3) * (-pressure_delta).clamp_min(0) / RHO_L)
    return vaporize - collapse


def multiphase_pinn_residuals(pred, inp, mask, nut, spacing, mean, std):
    next_state = pred * std[None, :, None, None] + mean[None, :, None, None]
    current = inp[:, :4] * std[None, :, None, None] + mean[None, :, None, None]
    u, v, p, alpha = next_state.unbind(1)
    u0, v0, _, alpha0 = current.unbind(1)
    alpha = alpha.clamp(0, 1); alpha0 = alpha0.clamp(0, 1)
    rho = (1 - alpha) * RHO_L + alpha * RHO_V
    rho0 = (1 - alpha0) * RHO_L + alpha0 * RHO_V
    nu_mix = (1 - alpha) * NU_L + alpha * NU_V
    dx, dy, dt = spacing[:, 0], spacing[:, 1], spacing[:, 2]
    fluid = mask[:, 0]
    valid = (fluid[:, 1:-1, 1:-1] * fluid[:, 1:-1, 2:] * fluid[:, 1:-1, :-2]
             * fluid[:, 2:, 1:-1] * fluid[:, :-2, 1:-1])

    div = ddx(u, dx) + ddy(v, dy)
    rho_c = rho[:, 1:-1, 1:-1]
    u_c, v_c = u[:, 1:-1, 1:-1], v[:, 1:-1, 1:-1]
    du_dt = (rho_c * u_c - rho0[:, 1:-1, 1:-1] * u0[:, 1:-1, 1:-1]) / dt[:, None, None]
    dv_dt = (rho_c * v_c - rho0[:, 1:-1, 1:-1] * v0[:, 1:-1, 1:-1]) / dt[:, None, None]
    mom_x = du_dt + ddx(rho * u * u, dx) + ddy(rho * u * v, dy) + ddx(p, dx)
    mom_y = dv_dt + ddx(rho * u * v, dx) + ddy(rho * v * v, dy) + ddy(p, dy)
    mu_eff = rho_c * (nu_mix[:, 1:-1, 1:-1] + nut[:, 0, 1:-1, 1:-1].clamp_min(0))
    mom_x = mom_x - mu_eff * laplacian(u, dx, dy)
    mom_y = mom_y - mu_eff * laplacian(v, dx, dy)
    phase = (alpha[:, 1:-1, 1:-1] - alpha0[:, 1:-1, 1:-1]) / dt[:, None, None]
    phase = phase + ddx(alpha * u, dx) + ddy(alpha * v, dy) - schnerr_sauer_alpha_source(alpha[:, 1:-1, 1:-1], p[:, 1:-1, 1:-1])

    def mse(residual, scale):
        normalized = torch.nan_to_num(residual / scale, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100, 100)
        return (normalized ** 2 * valid).sum() / valid.sum().clamp_min(1)

    return {
        "continuity": mse(div, 100.0),
        "x_momentum": mse(mom_x, 1.0e6),
        "y_momentum": mse(mom_y, 1.0e6),
        "vapor_transport": mse(phase, 1000.0),
    }


def multiphase_pinn_loss(pred, inp, mask, nut, spacing, mean, std):
    return sum(multiphase_pinn_residuals(pred, inp, mask, nut, spacing, mean, std).values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["unet", "fno", "deeponet", "pinn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    case_dir = ROOT / "data" / "processed" / "cases"
    mean, std = compute_stats(case_dir)
    (ROOT / "training_runs").mkdir(exist_ok=True)
    (ROOT / "data" / "processed" / "normalization.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2)+"\n")
    train_ds = CavitationDataset(case_dir, "train", mean, std)
    val_ds = CavitationDataset(case_dir, "val", mean, std)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mean_t = torch.tensor(mean, device=device); std_t = torch.tensor(std, device=device)
    for model_name in args.models:
        current_train_ds = CavitationPhysicsDataset(case_dir, "train", mean, std) if model_name == "pinn" else train_ds
        for seed in args.seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = build_model(model_name).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
            train_loader = DataLoader(current_train_ds, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size)
            history, best, best_state, stale_epochs = [], float("inf"), None, 0
            started = time.perf_counter()
            for epoch in range(args.epochs):
                model.train(); total = 0.0
                for batch in train_loader:
                    inp, target, mask = batch[:3]
                    inp, target, mask = inp.to(device), target.to(device), mask.to(device)
                    pred = model(inp)
                    loss = masked_mse(pred, target, mask)
                    if model_name == "pinn":
                        nut, spacing = batch[3].to(device), batch[4].to(device)
                        loss = loss + 1e-3 * multiphase_pinn_loss(pred, inp, mask, nut, spacing, mean_t, std_t)
                    optimizer.zero_grad(); loss.backward()
                    if model_name == "pinn":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); total += loss.item()*len(inp)
                model.eval(); val = 0.0
                with torch.no_grad():
                    for inp, target, mask in val_loader:
                        inp, target, mask = inp.to(device), target.to(device), mask.to(device)
                        val += masked_mse(model(inp), target, mask).item()*len(inp)
                record = {"epoch": epoch+1, "train_loss": total/len(current_train_ds), "val_loss": val/len(val_ds)}
                history.append(record)
                if record["val_loss"] < best:
                    best = record["val_loss"]
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                print(f"[{model_name} seed={seed}] {epoch+1:02d} train={record['train_loss']:.5g} val={record['val_loss']:.5g}", flush=True)
                if epoch + 1 >= 6 and stale_epochs >= args.patience:
                    print(f"[{model_name} seed={seed}] early stop at epoch {epoch+1}", flush=True)
                    break
            run_dir = ROOT / "training_runs" / f"{model_name}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": best_state, "mean": mean, "std": std, "model_name": model_name}, run_dir / "best.pt")
            (run_dir / "history.json").write_text(json.dumps({
                "seconds": time.perf_counter()-started,
                "max_epochs": args.epochs,
                "patience": args.patience,
                "best_epoch": min(history, key=lambda item: item["val_loss"])["epoch"],
                "history": history,
            }, indent=2)+"\n")


if __name__ == "__main__": main()
