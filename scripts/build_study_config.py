#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
families = ("NACA0012", "NACA2412", "NACA4412", "NACA0015", "NACA2415")
angles = (6.0, 9.0)
sigmas = (0.8, 1.2, 1.8)

cases = []
for family in families:
    for angle in angles:
        for sigma in sigmas:
            cases.append({
                "id": f"{family.lower()}_a{int(angle):02d}_s{int(round(100 * sigma)):03d}",
                "geometry": family,
                "chord_m": 0.1,
                "angle_deg": angle,
                "reynolds": 1_000_000.0,
                "cavitation_number": sigma,
                "end_time_s": 0.03,
                "write_interval_s": 0.001,
                "initial_delta_t_s": 1e-8,
                "mesh_size_foil_m": 0.001,
                "mesh_size_far_m": 0.02,
            })

output = ROOT / "configs" / "study.json"
output.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
print(f"Wrote {len(cases)} cases to {output}")
