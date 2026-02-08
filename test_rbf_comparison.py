"""
Test and compare RBF y-space pricer against standard Kim-Jang implementation.
"""

import numpy as np
from american_options.kim_jang_iter import KimJangIterative
from american_options.rbf_y_space import RBFYSpaceAmerican
from american_options.ju_1998 import Ju1998Pricing

# Test parameters
K = 100
r = 0.05
T = 0.5
sigma = 0.2
alpha = 0.0
S_test = [90, 95, 100, 105, 110]

print("=" * 80)
print("American Put Option Pricer Comparison")
print("=" * 80)
print(f"Parameters: K={K}, r={r}, T={T}, sigma={sigma}, alpha={alpha}")
print()

# Ju (1998) - closed form approximation
print("Ju (1998) - Closed Form:")
ju_pricer = Ju1998Pricing(100, K, r, T, sigma, alpha, "put")
for S in S_test:
    price = Ju1998Pricing(S, K, r, T, sigma, alpha, "put").price()
    print(f"  S={S:3d}: {price:.8f}")
print()

# Kim-Jang iterative
print("Kim-Jang Iterative:")
kim_pricer = KimJangIterative(K=K, r=r, T=T, sigma=sigma, alpha=alpha, option_type="put")
tau_kim, B_kim = kim_pricer.compute_boundary(n_nodes=20, n_iter=10)
for S in S_test:
    price = kim_pricer.price_put(S, T, boundary=(tau_kim, B_kim))
    print(f"  S={S:3d}: {price:.8f}")
print()
print(f"Kim boundary at T: {B_kim[-1]:.6f}")
print(f"Kim boundary at T/2: {np.interp(T/2, tau_kim, B_kim):.6f}")
print()

# RBF y-space
print("RBF Y-Space:")
rbf_pricer = RBFYSpaceAmerican(K=K, r=r, T=T, sigma=sigma, alpha=alpha, option_type="put")
y_knots, B_knots, rbf_interp = rbf_pricer.compute_boundary_rbf(n_knots=20, n_iter=20, verbose=False)
for S in S_test:
    price = rbf_pricer.price_put(S, T, boundary_rbf=(y_knots, B_knots, rbf_interp))
    print(f"  S={S:3d}: {price:.8f}")
print()
print(f"RBF boundary at y=y_max (tau=0): {B_knots[-1]:.6f}")
print(f"RBF boundary at y=sqrt(T/2): {rbf_interp(np.sqrt(T/2)):.6f}")
print()

# Boundary comparison
print("Boundary Comparison (tau vs y):")
print(f"{'Tau':<10} {'Kim B(tau)':<15} {'y=sqrt(tau)':<15} {'RBF B(y)':<15}")
for tau_val in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    B_kim_val = np.interp(tau_val, tau_kim, B_kim)
    y_val = np.sqrt(tau_val) if tau_val > 0 else 0
    B_rbf_val = rbf_interp(y_val)
    print(f"{tau_val:<10.3f} {B_kim_val:<15.6f} {y_val:<15.6f} {B_rbf_val:<15.6f}")
