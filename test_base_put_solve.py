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

# Solve to convergence
boundary_solved, hist = engine.solve_boundary(max_iters=50)

print(f"Base engine PUT:")
print(f"Boundary at T: {boundary_solved[0]:.2f}")
print(f"Boundary at t=0.9: {boundary_solved[2]:.2f}")
print(f"Boundary at t=0.5: {boundary_solved[-1]:.2f}")
print()

# Price
S0 = 100
results = engine.price(S0)
print(f"American: {results['amer']:.4f}")
