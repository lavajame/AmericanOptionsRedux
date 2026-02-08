import numpy as np
from new_american import AmericanRBFEngine, JuWarmStart

# Test parameters (PUT without dividends)
K = 100
T = 1.0
r = 0.05
q = 0.0
sigma = 0.25
w = -1  # PUT

# Create base engine
engine = AmericanRBFEngine(K=K, T=T, r=r, q=q, sigma=sigma, w=w, n_knots=16, alpha=1e-11)

# Get initial boundary from Ju
H = JuWarmStart.get_initial_H(engine)
B = np.concatenate([[engine.B0], np.exp(H)])

res, jac = engine.get_residual_and_jac(H)

print("Base engine (PUT, no dividends):")
print(f"B values: min={B.min():.2f}, max={B.max():.2f}")
print(f"Residual norm: {np.linalg.norm(res):.6f}")
print(f"Residual range: min={res.min():.6f}, max={res.max():.6f}")
print(f"Residual mean: {res.mean():.6f}")
