import numpy as np
from scipy.stats import norm
from new_american import AmericanRBFEngine, JuWarmStart

# Setup
S0, K, T, r, q, sigma, w, n_knots = 100, 100, 1.0, 0.05, -0.01, 0.10, +1, 10
pricer = AmericanRBFEngine(K=K, T=T, r=r, q=q, sigma=sigma, w=w, n_knots=n_knots)

# Get initial H
H = JuWarmStart.get_initial_H(pricer)
B = np.concatenate([[pricer.B0], np.exp(H)])

print("Testing log-space transformation")
print("="*50)

# Get the linear-space jacobian
res, jac_log = pricer.get_residual_and_jac(H)

# Also compute the linear jacobian manually
B = np.concatenate([[pricer.B0], np.exp(H)])
N = len(B)

B_spot = B[:, np.newaxis]
B_strike = B[np.newaxis, :]
tau_mat = np.maximum(pricer.y[:, np.newaxis]**2 - pricer.y[np.newaxis, :]**2, 1e-10)
mask = (pricer.y[:, np.newaxis]**2 - pricer.y[np.newaxis, :]**2) > 1e-12

f_mat, d1, exp_q = pricer._eep_integrand(B_spot, B_strike, tau_mat, pricer.y[np.newaxis, :])
euro_p, euro_d = pricer._black_scholes(B, pricer.K, pricer.y**2)

# Derivatives
t1 = (-2 * pricer.y[np.newaxis, :] * B_spot * exp_q * norm.pdf(d1)) / (B_strike * pricer.sigma * np.sqrt(tau_mat))
t2 = pricer.q - (pricer.r * pricer.K / B_strike) * np.exp(-(pricer.r - pricer.q) * tau_mat)
df_dB_strike = np.where(mask, t1 * t2, 0)

df_dB_spot = 2 * pricer.y[np.newaxis, :] * pricer.w * pricer.q * exp_q * norm.cdf(pricer.w * d1)

# Build linear Jacobian
J_lin = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        term_strike = -pricer.A[i, j] * df_dB_strike[i, j]
        term_spot = 0
        if i == j:
            term_spot = -np.sum(pricer.A[i, :] * df_dB_spot[i, :])
        term_exercise = 0
        if i == j:
            term_exercise = pricer.w - euro_d[i]
        J_lin[i, j] = term_strike + term_spot + term_exercise

# Test element [5,3]
i, j = 5, 3
print(f"\nElement J[{i},{j}]:")
print(f"Linear Jacobian (dres/dB): {J_lin[i,j]:.6f}")
print(f"Log Jacobian (dres/dH): {jac_log[i-1,j-1]:.6f}")
print(f"Expected transform: J_log = J_lin * B[j] = {J_lin[i,j] * B[j]:.6f}")

# Now do FD check in both spaces
epsilon = 1e-8

# FD in B-space
def get_res_at_B(B_test):
    B_spot = B_test[:, np.newaxis]
    B_strike = B_test[np.newaxis, :]
    tau_mat = np.maximum(pricer.y[:, np.newaxis]**2 - pricer.y[np.newaxis, :]**2, 1e-10)
    f_mat, _, _ = pricer._eep_integrand(B_spot, B_strike, tau_mat, pricer.y[np.newaxis, :])
    euro_p, _ = pricer._black_scholes(B_test, pricer.K, pricer.y**2)
    eep = np.diag(pricer.A @ f_mat.T)
    return (pricer.w * (B_test - pricer.K) - (euro_p + eep))[1:]

res_base_B = get_res_at_B(B)
B_bump = B.copy()
B_bump[j] += epsilon
res_bump_B = get_res_at_B(B_bump)
jac_fd_linear = (res_bump_B[i-1] - res_base_B[i-1]) / epsilon

print(f"\nFD in B-space: {jac_fd_linear:.6f}")
print(f"Error vs analytical: {abs(J_lin[i,j] - jac_fd_linear):.6e}")

# FD in H-space
def get_res_at_H(H_test):
    B_test = np.concatenate([[pricer.B0], np.exp(H_test)])
    return get_res_at_B(B_test)

res_base_H = get_res_at_H(H)
H_bump = H.copy()
H_bump[j-1] += epsilon
res_bump_H = get_res_at_H(H_bump)
jac_fd_log = (res_bump_H[i-1] - res_base_H[i-1]) / epsilon

print(f"\nFD in H-space: {jac_fd_log:.6f}")  
print(f"Expected (FD_B * B[j]): {jac_fd_linear * B[j]:.6f}")
print(f"Error vs analytical log jac: {abs(jac_log[i-1,j-1] - jac_fd_log):.6e}")
