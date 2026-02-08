import numpy as np
from new_american import AmericanRBFDividendEngine

# Test parameters (PUT with dividends)
K = 100
T = 1.0
r = 0.05
q = 0.0
sigma = 0.25
w = -1  # PUT
dividends = [(0.25, 3.0), (0.75, 3.0)]

# Create engine
engine = AmericanRBFDividendEngine(K=K, T=T, r=r, q=q, sigma=sigma, w=w, dividends=dividends, n_knots=16, alpha=1e-11)

# Test with boundary at strike (intrinsic value satisfied at all points for puts)
H_test = np.log(np.ones(engine.N - 1) * K * 0.85)  # Boundary at 85% of K
B_test = np.concatenate([[engine.B0], np.exp(H_test)])

res, jac = engine.get_residual_and_jac(H_test)

print("Testing with boundary at 0.85*K:")
print(f"B values: min={B_test.min():.2f}, max={B_test.max():.2f}")
print(f"Residual norm: {np.linalg.norm(res):.6f}")
print(f"Residual range: min={res.min():.6f}, max={res.max():.6f}")
print(f"Residual mean: {res.mean():.6f}")
print()
print("First 5 residuals:")
for i in range(min(5, len(res))):
    print(f"  res[{i}] = {res[i]:.6f}")
print()

# Try at intrinsic boundary
H_intr = np.log(np.ones(engine.N - 1) * K)  # Boundary at K
B_intr = np.concatenate([[engine.B0], np.exp(H_intr)])

res_intr, _ = engine.get_residual_and_jac(H_intr)

print("Testing with boundary at K (intrinsic):")
print(f"Residual norm: {np.linalg.norm(res_intr):.6f}")
print(f"Residual range: min={res_intr.min():.6f}, max={res_intr.max():.6f}")
