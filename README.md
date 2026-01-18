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

### Key scripts
- **compare_ju_fdm.py** — Run price comparisons across methods and generate EEB plots.
- **boundary_estimator_piecewise.py** — Boundary fitting logic (B-splines + piecewise \(\sqrt{\tau}\) polynomials), residuals, Jacobians, and pricing from fitted boundaries.
- **american_put_gk.py**, **gauss_kronrod.py** — GK boundary and pricing routines.
- **ju_1998.py**, **ju_1998_piecewise.py** — Ju closed-form approximations.
- **kim_jang_iter.py** — Iterative EEB solver.
- **fd_pricer.py** — Finite-difference pricer for benchmarking.
- **boundary_plotting.py** — Plotting utilities and EEB visualization.
- **run_discrete_div_demo.py** — Demo for discrete dividend handling.
- **verify_ju.py**, **verify_ju_piecewise.py** — Sanity checks against published results.

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

## Quick start

Run the main comparison (prices + boundary plots):

```bash
python compare_ju_fdm.py
```

This prints a comparison table across methods and writes plots to the repo root:
- `eeb_comparison_2.png`
- `call_eeb_comparison.png`

## More examples

Validate Ju approximation:

```bash
python verify_ju.py
python verify_ju_piecewise.py
```

Run the discrete dividend demo:

```bash
python run_discrete_div_demo.py
```

## Notes on the spline EEB fit

The spline fit uses a non-uniform time grid clustered near maturity and power-clustered knots in \(\tau = T - t\). A \(\sqrt{\tau}\) basis term can be included to capture the asymptotic behavior of the boundary near expiry. See `price_bspline_eeb()` and `boundary_bspline_eeb()` in compare_ju_fdm.py for the current configuration.

## Citation

If you use this code in academic or applied research, please cite the original references (Ju 1998/1999 and related literature) and this repository.
