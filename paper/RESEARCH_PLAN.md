# Benchmarking Neural Surrogates for Transient Hydrofoil Cavitation Prediction

## Primary question

How accurately and efficiently do neural surrogate architectures predict
transient cavitating flow around hydrofoils at unseen operating conditions and
geometries?

## New contribution relative to the prior work

The earlier study used steady, single-phase RANS fields and pressure-threshold
cavitation risk in a shape-optimization loop. This study uses newly generated
transient multiphase fields and evaluates actual vapor fraction, cavity
morphology, shedding dynamics, cavitating loads, rollout stability, and
computational cost. Shape optimization is outside the scope of this paper.

## Planned models

1. Persistence baseline
2. Autoregressive CNN-U-Net
3. Autoregressive FNO
4. Autoregressive DeepONet
5. Transient multiphase PINN with continuity, mixture-momentum, and
   Schnerr-Sauer vapor-transport residuals

## Required metrics

- Pressure and velocity errors
- Vapor-fraction MAE and R2
- Cavity-mask Dice and intersection over union
- Cavity length and total vapor-volume error
- Shedding frequency and Strouhal-number error
- Mean and fluctuating lift and drag
- Error versus rollout horizon
- Training time, inference time, and CFD speedup
- Three-seed mean and sample standard deviation

## Validation gates

- Mesh and time-step convergence for selected cases
- CFD comparison with published cavity length, pressure, and shedding data
- Entire trajectories kept within one split
- Held-out operating conditions and an untouched hydrofoil geometry
- No claims based on pressure-threshold risk
