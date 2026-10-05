#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
rows = []
for case in sorted((ROOT/"data/raw").glob("naca*")):
    log = case/"log.interPhaseChangeFoam"
    text = log.read_text() if log.exists() else ""
    mins = [float(x) for x in re.findall(r"Min\(alpha\.water\) = ([^ ]+)", text)]
    maxs = [float(x) for x in re.findall(r"Max\(alpha\.water\) = ([^\n ]+)", text)]
    times = [float(x) for x in re.findall(r"^Time = ([^\n ]+)", text, re.MULTILINE)]
    row = {
        "case": case.name,
        "complete": bool(re.search(r"^End$", text, re.MULTILINE)),
        "final_time": max(times, default=0),
        "alpha_water_min": min(mins, default=float("nan")),
        "alpha_water_max": max(maxs, default=float("nan")),
        "mesh_ok": "Mesh OK." in (case/"log.checkMesh").read_text(),
    }
    row["valid"] = row["complete"] and row["final_time"] >= 0.03 and row["alpha_water_min"] >= -1e-4 and row["alpha_water_max"] <= 1+1e-4 and row["mesh_ok"]
    rows.append(row)
(ROOT/"results").mkdir(exist_ok=True)
(ROOT/"results/cfd_audit.json").write_text(json.dumps(rows, indent=2)+"\n")
print(f"valid={sum(r['valid'] for r in rows)}/{len(rows)}")
for row in rows:
    if not row["valid"]: print(row)
