# Transient Hydrofoil Cavitation Dataset

This release accompanies the paper *Benchmarking Neural Surrogates for
Transient Hydrofoil Cavitation Forecasting*.

Permanent identifier: https://doi.org/10.5281/zenodo.23148648

## Contents

- `hydrofoil_cavitation_processed_v1.zip`: the 128 x 64 NumPy archives used by
  the training and evaluation scripts, together with normalization metadata.
- `hydrofoil_cavitation_openfoam_v1.tar.gz`: raw OpenFOAM v2312 case folders,
  solver outputs, meshes, force histories, and case metadata for the 30 study
  trajectories and the reported convergence checks.
- `SHA256SUMS`: SHA-256 checksums for both archives.

## Study matrix

The production dataset contains five NACA four-digit families (0012, 0015,
2412, 2415, and 4412), angles of attack 6 and 9 degrees, and cavitation numbers
0.8, 1.2, and 1.8. The family split used in the paper is:

- training: NACA 0012, 2412, and 4412;
- validation: NACA 0015; and
- final test: NACA 2415.

Each trajectory was generated with `interPhaseChangeFoam` using the
Schnerr-Sauer cavitation model, liquid water at Reynolds number 1,000,000, and
fields written every 0.001 s. The processed archives contain streamwise and
transverse velocity, pressure, vapor fraction, and auxiliary fields described
by the repository's extraction code.

## Reproduction

Code, configurations, model definitions, and evaluation scripts are available
at https://github.com/aaravvd/transient-hydrofoil-cavitation

Run the commands documented in that directory's `README.md`. The raw archive
is not required to train from the processed arrays, but it is provided for CFD
and preprocessing audits.

## Scope

The data are computational results, not experimental measurements. The paper's
grid study identifies non-negligible force and cavity-area sensitivity, so the
dataset should be interpreted as one documented CFD reference rather than
grid-independent physical truth.
