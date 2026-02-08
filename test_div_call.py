import numpy as np
from new_american import AmericanRBFDividendEngine

# Test parameters (CALL with dividends)
K = 100
T = 1.0
r = 0.05
q = 0.0
sigma = 0.25
w = 1  # CALL
dividends = [(0.25, 3.0), (0.75, 3.0)]

# Create engine
engine = AmericanRBFDividendEngine(K=K, T=T, r=r, q=q, sigma=sigma, w=w, dividends=dividends, n_knots=16, alpha=1e-11)

# Solve
boundary_solved, hist = engine.solve_boundary(max_iters=50)

print(f"Dividend Call:")
print(f"Boundary at T: {boundary_solved[0]:.2f}")
print(f"Boundary at t=0.9 (before first div at 0.75): {boundary_solved[2]:.2f}")
print(f"Boundary at t=0.5 (after both divs): {boundary_solved[-1]:.2f}")
print()

# Price  
S0 = 100
results = engine.price(S0)
print(f"American: {results['amer']:.4f}")
