# Transient Hydrofoil Cavitation Surrogate Benchmark

This repository contains the reproducible code, data-generation configuration,
trained checkpoints, metrics, figures, and manuscript for an independent study
of transient multiphase hydrofoil cavitation. It does not reuse the steady,
single-phase results or shape-optimization experiments from the earlier work.

## Study design

- Solver: OpenFOAM v2312 `interPhaseChangeFoam`
- Cavitation model: Schnerr-Sauer mass transfer
- Fluid: water, `rho = 997 kg/m^3`, `nu = 1e-6 m^2/s`
- Operating point: `Re = 1e6`, angles of attack 6 and 9 degrees,
  cavitation numbers 0.8, 1.2, and 1.8
- Training families: NACA 0012, 2412, and 4412
- Validation family: NACA 0015
- Final-test family: NACA 2415
- Models: CNN-U-Net, FNO, DeepONet, and a transient multiphase PINN
- Baseline: persistence
- Repetitions: seeds 7, 17, and 29

Each network maps one 128 x 64 flow snapshot to the next snapshot. Targets are
streamwise velocity, transverse velocity, pressure, and vapor volume fraction.
The final test set is an untouched geometry family.

## Data

The processed training arrays and raw OpenFOAM cases are archived on Zenodo:

- DOI: https://doi.org/10.5281/zenodo.23148648

The Zenodo record contains separate archives for the processed 128 x 64 arrays
and raw CFD cases, along with SHA-256 checksums and dataset documentation.

Repository: https://github.com/aaravvd/transient-hydrofoil-cavitation

## Reproduce the pipeline

Run commands from this directory. Dataset generation requires Docker, Gmsh,
and the `microfluidica/openfoam:2312` image. Model scripts require Python,
NumPy, SciPy, PyTorch, and Matplotlib.

```bash
python scripts/generate_cases.py --config configs/study.json
python scripts/run_cases.py --config configs/study.json
python scripts/audit_cases.py --config configs/study.json
python scripts/extract_dataset.py --config configs/study.json
python scripts/train_models.py --epochs 20 --patience 5 --seeds 7 17 29
python scripts/evaluate_models.py
python scripts/evaluate_rollouts.py
python scripts/evaluate_horizons.py
python scripts/evaluate_conditions.py
python scripts/evaluate_physics.py
python scripts/make_results.py
```

Convergence cases are defined in `configs/convergence.json` and
`configs/convergence_xfine.json`. 

## Important interpretation

The CFD reference remains spatially sensitive: the fine-to-extra-fine changes
are approximately 6% in mean lift, 4.8% in mean drag, and 4.8% in normalized
cavity area for the checked case. Reported surrogate errors therefore measure
agreement with this numerical reference, not experimental truth.

The PINN loss combines supervised next-state error with continuity,
conservative mixture-momentum, and vapor-transport residuals, which are also
reported separately on the final-test family. The
Schnerr-Sauer phase source is evaluated from predicted pressure and vapor
fraction. Saved OpenFOAM RANS eddy viscosity is privileged CFD information used
only while evaluating the PINN residual during training. It is neither an input
to the other models nor required by the PINN at inference.
