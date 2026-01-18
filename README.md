# American Options — Early Exercise Boundary (EEB) Toolkit

This repository contains research-grade Python implementations for American option pricing and early exercise boundary (EEB) estimation. It compares multiple approaches (closed-form approximations, iterative boundary solvers, Gauss–Kronrod quadrature, finite-difference benchmarks, and spline-based fits) and includes scripts to reproduce benchmark plots and tables.

## What’s inside

### Core approaches
- **Ju (1998/1999) closed-form EEB approximation** for American calls and puts with continuous dividends.
- **Kim–Jang iterative EEB solver** using integral equation iteration.
- **Gauss–Kronrod (GK) boundary/price solver** for stable numerical integration.
- **Finite-difference (FDM) benchmark** data for boundary and prices (used for validation).
- **Spline EEB fitting** with B-splines (including optional \(\sqrt{\tau}\) basis term) and non-uniform time grids.
- **Piecewise polynomial boundary in \(\sqrt{\tau}\)** for event-driven modeling (e.g., dividends).

### Project layout
- **american_options/** — Core library (Ju, Kim–Jang, GK, FDM, spline boundary fitting).
- **scripts/** — Runnable scripts and demos.
- **docs/papers/** — Reference PDFs.
- **outputs/** — Generated plots (ignored by git).

### Key scripts
- **scripts/compare_ju_fdm.py** — Run price comparisons across methods and generate EEB plots.
- **scripts/boundary_plotting.py** — Plotting utilities and EEB visualization.
- **scripts/run_discrete_div_demo.py** — Demo for discrete dividend handling.
- **scripts/verify_ju.py**, **scripts/verify_ju_piecewise.py** — Sanity checks against published results.

### Data & outputs
- **boundary_output_fdm.csv**, **boundary_output_gk.csv** — Precomputed boundaries.
- **eeb_comparison.png**, **eeb_comparison_2.png**, **call_eeb_comparison.png** — Generated plots.
- **Ju_1999.pdf**, **ssrn-362.pdf** — Reference material.

## Requirements

Tested with Python 3.10+.

Install dependencies:

```bash
pip install numpy scipy pandas matplotlib
```

## How to use

Run commands from the repo root.

1) Install dependencies:

```bash
pip install numpy scipy pandas matplotlib
```

2) Run the main comparison (prices + boundary plots):

```bash
python -m scripts.compare_ju_fdm
```

Outputs are written to `outputs/`:
- `outputs/eeb_comparison_2.png`
- `outputs/call_eeb_comparison.png`

3) Validate Ju approximation:

```bash
python -m scripts.verify_ju
python -m scripts.verify_ju_piecewise
```

4) Run the discrete dividend demo:

```bash
python -m scripts.run_discrete_div_demo
```

## Quick start (legacy)

If you prefer direct script execution, you can also run:

```bash
python scripts/compare_ju_fdm.py
python scripts/verify_ju.py
python scripts/verify_ju_piecewise.py
python scripts/run_discrete_div_demo.py
```

## Notes on the spline EEB fit

The spline fit uses a non-uniform time grid clustered near maturity and power-clustered knots in \(\tau = T - t\). A \(\sqrt{\tau}\) basis term can be included to capture the asymptotic behavior of the boundary near expiry. See `price_bspline_eeb()` and `boundary_bspline_eeb()` in scripts/compare_ju_fdm.py for the current configuration.

## Citation

If you use this code in academic or applied research, please cite the original references (Ju 1998/1999 and related literature) and this repository.
