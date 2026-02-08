import numpy as np
from new_american import AmericanRBFDividendEngine

# Test parameters (PUT with dividends)
S0 = 100
K = 100
T = 1.0
r = 0.05
q = 0.0
sigma = 0.25
w = -1  # PUT
dividends = [(0.25, 3.0), (0.75, 3.0)]

# Create engine
engine = AmericanRBFDividendEngine(K=K, T=T, r=r, q=q, sigma=sigma, w=w, dividends=dividends, n_knots=16, alpha=1e-11)

# Get initial boundary
H_init = engine._get_initial_boundary_with_divs()
B_init = np.concatenate([[engine.B0], np.exp(H_init)])

print("Initial boundary guess:")
print(f"Min: {B_init.min():.2f}")
print(f"Max: {B_init.max():.2f}")
print(f"Mean: {B_init.mean():.2f}")
print()
print("First 10 values:")
for i in range(min(10, len(B_init))):
    tau_i = engine.y[i]**2 if i < len(engine.y) else 0
    t_i = engine.T - tau_i
    print(f"  t={t_i:.3f}, tau={tau_i:.3f}, B={B_init[i]:.2f}")

