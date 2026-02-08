import numpy as np
import matplotlib.pyplot as plt
from new_american import AmericanRBFDividendEngine

S0, K, T, r, q, sigma, w = 100, 100, 1.0, 0.05, 0.0, 0.25, 1
dividends = [(0.4, 3.0)]

pricer = AmericanRBFDividendEngine(K=K, T=T, r=r, q=q, sigma=sigma, 
                                   dividends=dividends, w=w, n_knots=31)

# Get initial boundary
H_init = pricer._get_initial_boundary_with_divs()
B_init = np.concatenate([[pricer.B0], np.exp(H_init)])

print("Initial boundary values:")
print(f"Times (tau): {pricer.y**2}")
print(f"Boundaries: {B_init}")
print(f"\nMin: {B_init.min():.4f}, Max: {B_init.max():.4f}")

# Plot initial boundary
plt.figure(figsize=(10, 6))
plt.plot(pricer.y**2, B_init, 'o-', label='Initial Boundary')
plt.axhline(y=K, color='k', linestyle='--', alpha=0.5, label='Strike')
for t_star, d in dividends:
    plt.axvline(x=t_star, color='r', linestyle=':', alpha=0.5, label=f'Div at t={t_star}')
plt.xlabel('Time to Maturity')
plt.ylabel('Boundary')
plt.title('Initial Boundary Guess with Dividends')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Check residual at initial guess
res, jac = pricer.get_residual_and_jac(H_init)
print(f"\nInitial residual norm: {np.linalg.norm(res):.6f}")
print(f"Initial residual: {res[:10]}")
print(f"Jacobian condition number: {np.linalg.cond(jac):.2e}")
