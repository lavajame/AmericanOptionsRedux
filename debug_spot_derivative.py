import numpy as np
from scipy.stats import norm
from new_american import AmericanRBFEngine, JuWarmStart

# Setup
S0, K, T, r, q, sigma, w, n_knots = 100, 100, 1.0, 0.05, -0.01, 0.10, +1, 10
pricer = AmericanRBFEngine(K=K, T=T, r=r, q=q, sigma=sigma, w=w, n_knots=n_knots)

# Get initial H
H = np.log(np.linspace(pricer.B0, pricer.K * 1.2, pricer.N)[1:])
B = np.concatenate([[pricer.B0], np.exp(H)])

# Test the spot derivative formula
i = 5  # Test row 5
k = 3  # Test how f[5,3] changes with B[5]

B_spot = B[i]
B_strike = B[k]
tau = pricer.y[i]**2 - pricer.y[k]**2
u = pricer.y[k]

print(f"Testing df/dB_spot for f[{i},{k}]")
print(f"B_spot = B[{i}] = {B_spot:.6f}")
print(f"B_strike = B[{k}] = {B_strike:.6f}")  
print(f"tau = {tau:.6f}")
print(f"u = {u:.6f}")

# Compute integrand
v_sqrt_tau = pricer.sigma * np.sqrt(abs(tau))
if tau > 1e-12:
    d1 = (np.log(B_spot/B_strike) + (pricer.r - pricer.q + 0.5*pricer.sigma**2)*tau) / v_sqrt_tau
    d2 = d1 - v_sqrt_tau
    exp_q, exp_r = np.exp(-pricer.q * tau), np.exp(-pricer.r * tau)
    
    f_val = 2 * u * pricer.w * (pricer.q * B_spot * exp_q * norm.cdf(pricer.w * d1) - 
                                pricer.r * pricer.K * exp_r * norm.cdf(pricer.w * d2))
    print(f"f = {f_val:.6f}")
    print(f"d1 = {d1:.6f}")
    
    # Analytical derivative with respect to B_spot
    # f = 2u*w*[q*S*e^{-qt}*N(wd1) - r*K*e^{-rt}*N(wd2)]
    # df/dS = 2u*w*[q*e^{-qt}*N(wd1) + q*S*e^{-qt}*n(wd1)*d(d1)/dS - r*K*e^{-rt}*n(wd2)*d(d2)/dS]
    # d(d1)/dS = 1/(S*sigma*sqrt(tau))
    # d(d2)/dS = d(d1)/dS
    
    df_dS_term1 = 2 * u * pricer.w * pricer.q * exp_q * norm.cdf(pricer.w * d1)
    
    dd1_dS = 1 / (B_spot * pricer.sigma * np.sqrt(tau))
    df_dS_term2 = 2 * u * pricer.w * pricer.q * B_spot * exp_q * norm.pdf(d1) * pricer.w * dd1_dS
    df_dS_term3 = -2 * u * pricer.w * pricer.r * pricer.K * exp_r * norm.pdf(d2) * pricer.w * dd1_dS
    
    df_dS_analytical = df_dS_term1 + df_dS_term2 + df_dS_term3
    
    print(f"\nAnalytical df/dB_spot:")
    print(f"  Term 1 (N term): {df_dS_term1:.6f}")
    print(f"  Term 2 (S*n term): {df_dS_term2:.6f}")  
    print(f"  Term 3 (K*n term): {df_dS_term3:.6f}")
    print(f"  Total: {df_dS_analytical:.6f}")
    
    # What the code currently computes
    df_dS_code = 2 * u * pricer.w * pricer.q * exp_q * norm.cdf(pricer.w * d1)
    print(f"\nCode's df/dB_spot: {df_dS_code:.6f}")
    print(f"Missing terms: {df_dS_analytical - df_dS_code:.6f}")
    
    # FD check
    epsilon = 1e-8
    B_spot_bump = B_spot + epsilon
    d1_bump = (np.log(B_spot_bump/B_strike) + (pricer.r - pricer.q + 0.5*pricer.sigma**2)*tau) / v_sqrt_tau
    d2_bump = d1_bump - v_sqrt_tau
    f_bump = 2 * u * pricer.w * (pricer.q * B_spot_bump * exp_q * norm.cdf(pricer.w * d1_bump) - 
                                 pricer.r * pricer.K * exp_r * norm.cdf(pricer.w * d2_bump))
    df_dS_fd = (f_bump - f_val) / epsilon
    
    print(f"\nFD df/dB_spot: {df_dS_fd:.6f}")
    print(f"Analytical error: {abs(df_dS_analytical - df_dS_fd):.6e}")
    print(f"Code error: {abs(df_dS_code - df_dS_fd):.6e}")
else:
    print("tau <= 0, skipping...")
