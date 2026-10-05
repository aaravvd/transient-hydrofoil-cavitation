#!/usr/bin/env python3
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = []
for path in sorted((ROOT/"data/processed/cases").glob("conv_*.npz")):
    data = np.load(path)
    fields = data["fields"][-6:]
    dx, dy = float(np.diff(data["x"]).mean()), float(np.diff(data["y"]).mean())
    force_path = ROOT/"data/raw"/path.stem/"postProcessing/forceCoeffs/0/coefficient.dat"
    force = np.loadtxt(force_path, comments="#")
    recent = force[force[:,0] >= 0.024]
    mesh_log = (ROOT/"data/raw"/path.stem/"log.checkMesh").read_text()
    cells = int(re.search(r"cells:\s+(\d+)", mesh_log).group(1))
    rows.append({
        "case": path.stem, "cells": cells,
        "mean_cd": float(recent[:,1].mean()), "mean_cl": float(recent[:,4].mean()),
        "mean_cavity_area_over_c2": float(np.clip(fields[...,3],0,1).sum((1,2)).mean()*dx*dy/0.1**2),
        "mean_min_pressure_pa": float(fields[...,2].min((1,2)).mean()),
    })
(ROOT/"results/convergence.json").write_text(json.dumps(rows, indent=2)+"\n")
print(json.dumps(rows, indent=2))
