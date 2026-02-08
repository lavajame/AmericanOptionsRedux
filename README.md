# American Options Redux — Kim's Integral Equation in y-space

Research-grade Python implementation of Kim's integral equation approach for American option pricing with discrete dividends, using Gaussian RBF integration and Newton's method.

## What's inside

**Core implementation:**
- **Kim's integral equation in y-space** (y = √τ) with Gaussian RBF integration
- **Newton solver** with analytic lower-triangular Jacobian for simultaneous boundary solve
- **Design matrix representation** using asymptotic powers y^k/(c^k + y^k) + step functions at dividend times
- **Ju (1998) warm-start** for initial boundary guess
- **Discrete dividend support** with ex-dividend pricing framework
- **L2 ridge regression** for boundary smoothing and stability
- **Economic boundary constraints** to prevent irrational early exercise

**Key features:**
- ✅ Handles American calls and puts
- ✅ Continuous dividend yield (q)
- ✅ Discrete dividends at specified times
- ✅ Automatic dividend knot placement for boundary discontinuities
- ✅ Finite-difference Jacobian validation
- ✅ Residual diagnostics showing error concentration

## Files

- **kim_integral_rbf.py** — Main implementation with solver, pricer, and test harness
- **kim_integral_rbf_convergence_divs.png** — Convergence plot with discrete dividends
- **kim_integral_rbf_convergence_no_divs.png** — Convergence plot without dividends
- **fd_dividend_boundary.png** — Boundary comparison plot

## Requirements

Python 3.10+ with:

```bash
pip install numpy scipy pandas matplotlib
```

## Quick start

Run the included test harness:

```bash
python kim_integral_rbf.py
```

This will:
1. Price an American call with discrete dividend D=10.0 at t=0.4
2. Run Newton iterations to solve for the boundary
3. Validate the analytic Jacobian via finite differences
4. Generate convergence plots with residual diagnostics
5. Print detailed analysis of error concentration

**Example output:**
```
American Call  (S=100, K=100, T=1, r=5.0%, q=5.0%, sigma=25.0%)
Discrete dividends: 10.0 at t=0.40
============================================================
  Euro : 5.08747726
  EEP  : 0.12118963
  Amer : 5.20866688
  Time : 0.0974s

Jacobian Finite-Difference Test
  Max error  : 1.138554e-03

Final Residual Analysis
  Residual by region:
    Before: RMSE = 5.143e-03
    After:  RMSE = 1.940e-01
    Ratio:  37.7x worse after dividend
```

## Usage example

```python
from kim_integral_rbf import KimIntegralRBF

# American call with discrete dividend
pricer = KimIntegralRBF(
    K=100, T=1.0, r=0.05, q=0.05, sigma=0.25, 
    w=+1,  # +1 for call, -1 for put
    N=60,  # number of knots
    n_powers=1,  # asymptotic power terms
    div_times=[0.4],    # dividend at t=0.4
    div_amounts=[10.0]  # dividend amount
)

# Price at spot S0=100
result = pricer.price(S0=100, max_iters=10)

print(f"European: {result['euro']:.4f}")
print(f"EEP:      {result['eep']:.4f}")
print(f"American: {result['amer']:.4f}")
```

## Key implementation details

### Design matrix basis functions
- **Power terms:** y^k / (c^k + y^k) approach 1 as y→∞ (long-maturity asymptote)
- **Step functions:** H(y - y_d) at each dividend time to capture discontinuities
- **Ridge regression:** L2 penalty (λ=0.001) prevents noise chasing
- All basis functions = 0 at y=0 so B(0)=K automatically

### Ex-dividend pricing framework
- European: S_ex = S0 - PV(all divs), martingale under risk-neutral measure
- Boundary residual: Uses S_adj = B[k] - PV(divs from t_k to T) in BS formula
- **Critical fix:** EEP integrand uses S0 (dirty) for correct absolute value, but S_ex/B_ex ratio in d1 for correct drift

### Economic constraints
- Calls: K ≤ B ≤ 3K (only exercise ITM, limit irrational regions)
- Puts: 0.5K ≤ B ≤ K

## Known issues and future work

**Current limitation:** Residuals concentrate after dividend times (37x worse RMSE)
- Root cause: Step functions allow boundary jumps but don't provide slope flexibility
- **Suggested improvement:** Add linear ramps y·H(y - y_d) for each dividend
- This would allow different slopes before/after dividends
- See diagnostic output: "Consider y*(y>y_d) terms for post-dividend flexibility"

## Citation

If you use this code in research, please cite:
- Kim, I.J. (1990). "The Analytic Valuation of American Options." Review of Financial Studies.
- This repository: https://github.com/lavajame/AmericanOptionsRedux
