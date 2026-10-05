#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cavpaper.models import build_model

RESULTS = ROOT/"results"; FIGURES = ROOT/"paper/figures"; FIGURES.mkdir(parents=True, exist_ok=True)
NAMES = {"unet":"CNN-U-Net", "fno":"FNO", "deeponet":"DeepONet", "pinn":"PINN", "persistence":"Persistence"}
COLORS = {"CNN-U-Net":"#2878B5", "FNO":"#D9553B", "DeepONet":"#6B5B95", "PINN":"#2A9D62", "Persistence":"#666666"}


def aggregate(path):
    rows = json.loads(path.read_text()); output=[]
    for model in ("unet","fno","deeponet","pinn","persistence"):
        subset=[r for r in rows if r["model"]==model]
        record={"model":NAMES[model]}
        for key in subset[0]:
            if key in ("model","seed","parameters"): continue
            vals=np.asarray([r[key] for r in subset],float); record[key+"_mean"]=float(vals.mean()); record[key+"_std"]=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
        record["parameters"]=subset[0].get("parameters",0); output.append(record)
    return output


one=aggregate(RESULTS/"test_metrics.json"); rollout=aggregate(RESULTS/"rollout_metrics.json")
(RESULTS/"aggregate_one_step.json").write_text(json.dumps(one,indent=2)+"\n")
(RESULTS/"aggregate_rollout.json").write_text(json.dumps(rollout,indent=2)+"\n")
for filename, rows in (("aggregate_one_step.csv",one),("aggregate_rollout.csv",rollout)):
    with (RESULTS/filename).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

# Training curves.
fig,ax=plt.subplots(figsize=(5.7,3.3))
for model in ("unet","fno","deeponet","pinn"):
    curves=[]
    for path in sorted((ROOT/"training_runs").glob(f"{model}_seed*/history.json")):
        curves.append([r["val_loss"] for r in json.loads(path.read_text())["history"]])
    width=max(map(len,curves)); arr=np.full((len(curves),width),np.nan)
    for index,curve in enumerate(curves): arr[index,:len(curve)]=curve
    x=np.arange(1,width+1); label=NAMES[model]; avg=np.nanmean(arr,0); spread=np.nanstd(arr,0)
    ax.plot(x,avg,label=label,color=COLORS[label]); ax.fill_between(x,avg-spread,avg+spread,color=COLORS[label],alpha=.15)
ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss"); ax.grid(alpha=.2); ax.legend(ncol=2,fontsize=8)
fig.tight_layout(); fig.savefig(FIGURES/"training_curves.png",dpi=240); plt.close(fig)

# Metric comparison.
fig,axes=plt.subplots(1,3,figsize=(7.1,2.5),constrained_layout=True)
models=[r["model"] for r in one]; x=np.arange(len(models))
specs=[("p_rmse_mean","p_rmse_std","One-step pressure RMSE","Pa",one),("vapor_iou_mean","vapor_iou_std","One-step vapor IoU","",one),("vapor_iou_mean","vapor_iou_std","20-step rollout IoU","",rollout)]
for ax,(key,err,title,ylabel,rows) in zip(axes,specs):
    ax.bar(x,[r[key] for r in rows],yerr=[r[err] for r in rows],color=[COLORS[m] for m in models],capsize=2)
    ax.set_title(title,fontsize=9); ax.set_ylabel(ylabel); ax.set_xticks(x,models,rotation=35,ha="right",fontsize=7); ax.grid(axis="y",alpha=.2)
fig.savefig(FIGURES/"metric_comparison.png",dpi=240); plt.close(fig)

# Error growth across the autoregressive horizon.
horizon=json.loads((RESULTS/"horizon_metrics.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(6.9,2.55),constrained_layout=True)
for model in ("unet","fno","deeponet","pinn","persistence"):
    label=NAMES[model]; subset=[row for row in horizon if row["model"]==model]
    steps=sorted({row["step"] for row in subset}); means=[]; spreads=[]
    for step in steps:
        vals=np.asarray([row["vapor_iou"] for row in subset if row["step"]==step])
        means.append(vals.mean()); spreads.append(vals.std(ddof=1) if len(vals)>1 else 0)
    means=np.asarray(means); spreads=np.asarray(spreads)
    axes[0].plot(steps,means,label=label,color=COLORS[label]); axes[0].fill_between(steps,means-spreads,means+spreads,color=COLORS[label],alpha=.12)
    pmeans=[]; pspreads=[]
    for step in steps:
        vals=np.asarray([row["p_rmse"] for row in subset if row["step"]==step])/1000
        pmeans.append(vals.mean()); pspreads.append(vals.std(ddof=1) if len(vals)>1 else 0)
    pmeans=np.asarray(pmeans); pspreads=np.asarray(pspreads)
    axes[1].plot(steps,pmeans,label=label,color=COLORS[label]); axes[1].fill_between(steps,pmeans-pspreads,pmeans+pspreads,color=COLORS[label],alpha=.12)
axes[0].set_ylabel("Cavity IoU"); axes[1].set_ylabel("Pressure RMSE (kPa)")
for ax in axes: ax.set_xlabel("Autoregressive step"); ax.grid(alpha=.2)
axes[0].legend(ncol=2,fontsize=7)
fig.savefig(FIGURES/"horizon_curves.png",dpi=240); plt.close(fig)

# Representative unseen-family field comparison, seed 7.
case=np.load(ROOT/"data/processed/cases/naca2415_a09_s120.npz"); meta=json.loads(str(case["metadata"]))
stats=json.loads((ROOT/"data/processed/normalization.json").read_text()); mean=np.asarray(stats["mean"],np.float32); std=np.asarray(stats["std"],np.float32)
i=-2; current=(case["fields"][i]-mean)/std; xg,yg=np.meshgrid(case["x"]/meta["chord_m"],case["y"]/meta["chord_m"]); h,w=xg.shape
extra=np.stack([xg,yg,case["fluid_mask"],np.full((h,w),meta["angle_deg"]/10),np.full((h,w),meta["cavitation_number"]/2),np.full((h,w),float(case["times"][i])/0.03)],-1)
inp=torch.from_numpy(np.concatenate([current,extra],-1).transpose(2,0,1)[None].astype(np.float32))
fields=[case["fields"][i+1]]; labels=["OpenFOAM"]
for model in ("unet","fno","deeponet","pinn"):
    payload=torch.load(ROOT/f"training_runs/{model}_seed7/best.pt",map_location="cpu",weights_only=False); net=build_model(model); net.load_state_dict(payload["model"]); net.eval()
    with torch.no_grad(): pred=net(inp).numpy()[0].transpose(1,2,0)*std+mean
    fields.append(pred); labels.append(NAMES[model])
fig,axes=plt.subplots(2,5,figsize=(9.0,3.5),sharex=True,sharey=True,constrained_layout=True)
pmin=min(f[...,2].min() for f in fields)/1000; pmax=max(f[...,2].max() for f in fields)/1000
for col,(field,label) in enumerate(zip(fields,labels)):
    im=axes[0,col].pcolormesh(xg,yg,field[...,2]/1000,shading="auto",cmap="coolwarm",vmin=pmin,vmax=pmax); axes[0,col].set_title(label,fontsize=8)
    iv=axes[1,col].pcolormesh(xg,yg,np.clip(field[...,3],0,1),shading="auto",cmap="viridis",vmin=0,vmax=1)
    for ax in axes[:,col]: ax.set_aspect("equal")
axes[0,0].set_ylabel("Pressure\ny/c"); axes[1,0].set_ylabel(r"Vapor $\alpha_v$\ny/c")
for ax in axes[1]: ax.set_xlabel("x/c")
fig.colorbar(im,ax=axes[0],label="kPa",shrink=.75); fig.colorbar(iv,ax=axes[1],label=r"$\alpha_v$",shrink=.75)
fig.savefig(FIGURES/"field_comparison.png",dpi=240); plt.close(fig)

# Grid/time convergence.
conv=json.loads((RESULTS/"convergence.json").read_text()); order=["coarse","medium","fine","xfine"]
mesh=[next(r for r in conv if f"conv_{name}_" in r["case"]) for name in order]
fig,axes=plt.subplots(1,3,figsize=(7.0,2.35),constrained_layout=True)
for ax,key,title in zip(axes,("mean_cl","mean_cd","mean_cavity_area_over_c2"),(r"Mean $C_L$",r"Mean $C_D$",r"Cavity area/$c^2$")):
    positions=np.arange(len(mesh)); ax.plot(positions,[r[key] for r in mesh],"o-",color="#2878B5"); ax.set_title(title,fontsize=9); ax.set_xlabel("Grid"); ax.set_xticks(positions,[f"{name}\n{row['cells']:,}" for name,row in zip(order,mesh)],fontsize=6); ax.grid(alpha=.2)
fig.savefig(FIGURES/"convergence.png",dpi=240); plt.close(fig)

# CFD runtime and end-to-end speedup using measured batched surrogate timing.
cfd=[]
for log in (ROOT/"data/raw").glob("naca*/log.interPhaseChangeFoam"):
    values=re.findall(r"ClockTime = (\d+) s",log.read_text());
    if values: cfd.append(float(values[-1]))
runtime={"cfd_seconds_per_trajectory_mean":float(np.mean(cfd)),"cfd_seconds_per_trajectory_std":float(np.std(cfd,ddof=1))}
for row in one:
    if row["model"] != "Persistence": runtime[row["model"]+"_20_step_speedup"]=runtime["cfd_seconds_per_trajectory_mean"]/(20*row["inference_ms_per_frame_mean"]/1000)
(RESULTS/"runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
print(json.dumps({"one_step":one,"rollout":rollout,"runtime":runtime},indent=2))
