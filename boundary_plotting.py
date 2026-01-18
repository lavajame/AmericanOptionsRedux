import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erf

def calculate_boundary_corrected(K, sigma, r, q, T_array, w=-1):
    """
    K: Strike
    sigma: Vol
    r: Risk-free rate
    q: Dividend yield
    T_array: Time to maturity
    w: -1 for Put, +1 for Call
    """
    T = np.maximum(np.array(T_array), 1e-8)
    
    # Adjust parameters based on w
    # For a Call (w=1), we effectively swap r and q roles in the exercise logic
    r_eff = r if w == -1 else q
    q_eff = q if w == -1 else r
    
    # k definitions for the 0-th iteration (B=K)
    # This ensures d1 and d2 have the correct perspective
    k1 = (r - q + 0.5 * sigma**2) / sigma
    k2 = (r - q - 0.5 * sigma**2) / sigma
    
    # Kernel constants
    A1 = r + 0.5 * (k1**2)
    A2 = r + 0.5 * (k2**2)
    
    def integral_kernel(A, time):
        return np.sqrt(np.pi / A) * erf(np.sqrt(A * time))

    # Numerator (Cash-side components)
    term_num_1 = (K * np.exp(-r * T)) / (sigma * np.sqrt(2 * np.pi * T)) * np.exp(-0.5 * (k2**2) * T)
    term_num_2 = (r * K / (sigma * np.sqrt(2))) * integral_kernel(A2, T)
    num = term_num_1 + term_num_2
    
    # Denominator (Asset-side components)
    # Using 'w' to correctly select the ITM probability tail
    term_den_1 = np.exp(-q * T) * norm.cdf(w * k1 * np.sqrt(T))
    term_den_2 = (np.exp(-q * T) / (sigma * np.sqrt(2 * np.pi * T))) * np.exp(-0.5 * (k1**2) * T)
    
    # Numerical integration for the N() term
    steps = 100
    int_N = np.zeros_like(T)
    for i, t_max in enumerate(T):
        u = np.linspace(1e-8, t_max, steps)
        du = t_max / steps
        # The probability tail flips for calls
        int_N[i] = np.sum(np.exp(-r * u) * norm.cdf(w * k1 * np.sqrt(u))) * du
        
    term_den_3 = q * int_N
    term_den_4 = (q * r / (sigma * np.sqrt(2))) * integral_kernel(A1, T)
    
    den = term_den_1 + term_den_2 + term_den_3 + term_den_4
    
    # The first iteration boundary
    B = num / den
    
    # Sanity check: If the math produces a Put boundary > K or Call < K 
    # due to q/r imbalances, we cap it at K (the immediate exercise floor)
    if w == -1:
        return np.minimum(B, K)
    else:
        return np.maximum(B, K)

# --- Run and Plot ---
times = np.linspace(0.001, 1.0, 100)
b_put = calculate_boundary_corrected(100, 0.2, 0.05, 0.02, times, w=-1)
b_call = calculate_boundary_corrected(100, 0.2, 0.05, 0.07, times, w=1) # q > r for Call exercise

plt.plot(times, b_put, label="Put Boundary (B < K)")
plt.plot(times, b_call, label="Call Boundary (B > K)")
plt.axhline(100, color='black', linestyle='--')
plt.legend()
plt.show()