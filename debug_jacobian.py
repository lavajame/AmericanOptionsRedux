import numpy as np
from scipy.stats import norm
from new_american import AmericanRBFEngine

# Setup
S0, K, T, r, q, sigma, w, n_knots = 100, 100, 1.0, 0.05, -0.01, 0.10, +1, 10
pricer = AmericanRBFEngine(K=K, T=T, r=r, q=q, sigma=sigma, w=w, n_knots=n_knots)

# Get initial H from warm start
from new_american import JuWarmStart
H = JuWarmStart.get_initial_H(pricer)

# Manual check for a specific element
B = np.concatenate([[pricer.B0], np.exp(H)])
print(f"B values: {B}")
print(f"y values: {pricer.y}")

# Check the integrand and its derivative manually for specific i, j
i, j = 5, 3  # Check element [5,3]
print(f"\n=== Checking df_dB_strike[{i},{j}] ===")

B_spot_val = B[i]
B_strike_val = B[j]
tau_val = pricer.y[i]**2 - pricer.y[j]**2
u_val = pricer.y[j]

print(f"B_spot = B[{i}] = {B_spot_val:.6f}")
print(f"B_strike = B[{j}] = {B_strike_val:.6f}")
print(f"tau = {tau_val:.6f}")
print(f"u = {u_val:.6f}")

# Compute integrand at this point
v_sqrt_tau = pricer.sigma * np.sqrt(tau_val)
d1 = (np.log(B_spot_val/B_strike_val) + (pricer.r - pricer.q + 0.5*pricer.sigma**2)*tau_val) / v_sqrt_tau
d2 = d1 - v_sqrt_tau
exp_q, exp_r = np.exp(-pricer.q * tau_val), np.exp(-pricer.r * tau_val)

f_val = 2 * u_val * pricer.w * (pricer.q * B_spot_val * exp_q * norm.cdf(pricer.w * d1) - 
                                  pricer.r * pricer.K * exp_r * norm.cdf(pricer.w * d2))
print(f"f = {f_val:.6f}")
print(f"d1 = {d1:.6f}, d2 = {d2:.6f}")

# Analytical derivative
t1 = (-2 * u_val * B_spot_val * exp_q * norm.pdf(d1)) / (B_strike_val * pricer.sigma * np.sqrt(tau_val))
t2 = pricer.q - (pricer.r * pricer.K / B_strike_val) * np.exp(-(pricer.r - pricer.q) * tau_val)
df_analytical = t1 * t2
print(f"\nAnalytical df/dB_strike = {df_analytical:.6f}")

# Finite difference check
epsilon = 1e-8
B_strike_bumped = B_strike_val + epsilon
d1_bump = (np.log(B_spot_val/B_strike_bumped) + (pricer.r - pricer.q + 0.5*pricer.sigma**2)*tau_val) / v_sqrt_tau
d2_bump = d1_bump - v_sqrt_tau
f_bumped = 2 * u_val * pricer.w * (pricer.q * B_spot_val * exp_q * norm.cdf(pricer.w * d1_bump) - 
                                     pricer.r * pricer.K * exp_r * norm.cdf(pricer.w * d2_bump))
df_fd = (f_bumped - f_val) / epsilon
print(f"Finite Diff df/dB_strike = {df_fd:.6f}")
print(f"Error = {abs(df_analytical - df_fd):.6e}")

# Now check the full Jacobian element calculation
print(f"\n=== Jacobian Element J[{i},{j}] ===")
res, jac = pricer.get_residual_and_jac(H)
print(f"Analytical J[{i},{j}] = {jac[i-1, j-1]:.6f}")  # -1 because jacobian starts from index 1

# FD check on the Jacobian
def get_res_at_H(H_test):
    B_test = np.concatenate([[pricer.B0], np.exp(H_test)])
    B_spot = B_test[:, np.newaxis]
    B_strike = B_test[np.newaxis, :]
    tau_mat = np.maximum(pricer.y[:, np.newaxis]**2 - pricer.y[np.newaxis, :]**2, 1e-10)
    f_mat, _, _ = pricer._eep_integrand(B_spot, B_strike, tau_mat, pricer.y[np.newaxis, :])
    euro_p, _ = pricer._black_scholes(B_test, pricer.K, pricer.y**2)
    eep = np.diag(pricer.A @ f_mat.T)
    return (pricer.w * (B_test - pricer.K) - (euro_p + eep))[1:]

res_base = get_res_at_H(H)
H_bump = H.copy()
H_bump[j-1] += epsilon
res_bump = get_res_at_H(H_bump)
jac_fd = (res_bump[i-1] - res_base[i-1]) / epsilon * B[j]  # multiply by B[j] for log-space

print(f"FD J[{i},{j}] = {jac_fd:.6f}")
print(f"Error = {abs(jac[i-1, j-1] - jac_fd):.6e}")
