import numpy as np
from scipy.stats import norm
from new_american import AmericanRBFEngine, JuWarmStart

# Setup
S0, K, T, r, q, sigma, w, n_knots = 100, 100, 1.0, 0.05, -0.01, 0.10, +1, 10
pricer = AmericanRBFEngine(K=K, T=T, r=r, q=q, sigma=sigma, w=w, n_knots=n_knots)

# Get initial H
H = np.log(np.linspace(pricer.B0, pricer.K * 1.2, pricer.N)[1:])
B = np.concatenate([[pricer.B0], np.exp(H)])

print("Testing DIAGONAL element J[3,3]")
print("="*50)

# Get the jacobian
res, jac_log = pricer.get_residual_and_jac(H)

# Test element [3,3] (which in 0-indexed log jacobian is [2,2])
i, j = 3, 3
print(f"\nElement J[{i},{j}] (diagonal):")
print(f"Analytical Log Jacobian: {jac_log[i-1,j-1]:.6f}")

# FD in H-space
epsilon = 1e-8

def get_res_at_H(H_test):
    B_test = np.concatenate([[pricer.B0], np.exp(H_test)])
    B_spot = B_test[:, np.newaxis]
    B_strike = B_test[np.newaxis, :]
    tau_mat = np.maximum(pricer.y[:, np.newaxis]**2 - pricer.y[np.newaxis, :]**2, 1e-10)
    f_mat, _, _ = pricer._eep_integrand(B_spot, B_strike, tau_mat, pricer.y[np.newaxis, :])
    euro_p, _ = pricer._black_scholes(B_test, pricer.K, pricer.y**2)
    eep = np.diag(pricer.A @ f_mat.T)
    return (pricer.w * (B_test - pricer.K) - (euro_p + eep))[1:]

res_base_H = get_res_at_H(H)
H_bump = H.copy()
H_bump[j-1] += epsilon
res_bump_H = get_res_at_H(H_bump)
jac_fd_log = (res_bump_H[i-1] - res_base_H[i-1]) / epsilon

print(f"FD Log Jacobian: {jac_fd_log:.6f}")  
print(f"Error: {abs(jac_log[i-1,j-1] - jac_fd_log):.6e}")
print(f"Relative Error: {abs(jac_log[i-1,j-1] - jac_fd_log) / abs(jac_fd_log) * 100:.2f}%")

# Print the full column 3 comparison
print(f"\n\nFull Column {j} Comparison:")
print("Row | Analytical |    FD    | AbsError")
print("----|------------|----------|----------")
for row_idx in range(len(jac_log)):
    H_bump = H.copy()
    H_bump[j-1] += epsilon
    res_bump = get_res_at_H(H_bump)
    jac_fd_elem = (res_bump[row_idx] - res_base_H[row_idx]) / epsilon
    ana_elem = jac_log[row_idx, j-1]
    err = abs(ana_elem - jac_fd_elem)
    print(f" {row_idx}  | {ana_elem:10.6f} | {jac_fd_elem:10.6f} | {err:8.6f}")
