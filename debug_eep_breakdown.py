"""
Debug script to understand EEP calculation with and without dividends
"""
import numpy as np
from scipy.stats import norm
from kim_integral_rbf import KimIntegralRBF

# Test parameters
N = 20
max_iters = 5
n_powers = 3
r = 0.05
q = 0.05
sigma = 0.25

print('=' * 80)
print('CASE 1: NO DIVIDENDS')
print('=' * 80)
pricer_no_div = KimIntegralRBF(
    K=100, T=1.0, r=r, q=q, sigma=sigma, w=+1, N=N, n_powers=n_powers,
    div_times=[], div_amounts=[])
res_no_div = pricer_no_div.price(S0=100, max_iters=max_iters, verbose_diagnostics=False)

print(f'\nResults:')
print(f'  S0   : {100:.4f}')
print(f'  Euro : {res_no_div["euro"]:.8f}')
print(f'  EEP  : {res_no_div["eep"]:.8f}')
print(f'  Amer : {res_no_div["amer"]:.8f}')

# Manually compute EEP breakdown
S0 = 100
B_final = res_no_div['boundary']['B'].values
y = res_no_div['boundary']['y'].values
print(f'\nEEP Integrand Breakdown (first few knots):')
print(f'  {"j":<4} {"tau":<10} {"B(tau)":<12} {"S0/B":<10} {"d1":<10} {"integrand f":<15}')

f_vec = np.zeros(N)
for j in range(min(10, N)):
    tau_j = 1.0 - y[j]**2
    if tau_j < 1e-12:
        continue
    
    B_j = B_final[j]
    vt = sigma * np.sqrt(tau_j)
    d1 = (np.log(S0 / B_j) + (r - q + 0.5 * sigma**2) * tau_j) / vt
    d2 = d1 - vt
    
    # Call option (w=+1)
    f_j = 2 * y[j] * (
        q * S0 * np.exp(-q * tau_j) * norm.cdf(d1)
        - r * 100 * np.exp(-r * tau_j) * norm.cdf(d2)
    )
    f_vec[j] = f_j
    
    print(f'  {j:<4} {tau_j:<10.6f} {B_j:<12.4f} {S0/B_j:<10.6f} {d1:<10.4f} {f_j:<15.8f}')

print()
print('=' * 80)
print('CASE 2: WITH DISCRETE DIVIDEND D=5.0 at t=0.4')
print('=' * 80)
pricer_with_div = KimIntegralRBF(
    K=100, T=1.0, r=r, q=q, sigma=sigma, w=+1, N=N, n_powers=n_powers,
    div_times=[0.4], div_amounts=[5.0])
res_with_div = pricer_with_div.price(S0=100, max_iters=max_iters, verbose_diagnostics=False)

# Compute ex-dividend prices
pv_all = pricer_with_div._pv_divs(1.0)  # PV of all divs at t=0
S_ex = 100 - pv_all

print(f'\nResults:')
print(f'  S0      : {100:.4f} (dirty)')
print(f'  PV(divs): {pv_all:.4f}')
print(f'  S_ex    : {S_ex:.4f} (ex-dividend)')
print(f'  Euro    : {res_with_div["euro"]:.8f}')
print(f'  EEP     : {res_with_div["eep"]:.8f}')
print(f'  Amer    : {res_with_div["amer"]:.8f}')

# Manually compute EEP breakdown
B_dirty = res_with_div['boundary']['B'].values
y_div = res_with_div['boundary']['y'].values

print(f'\nEEP Integrand Breakdown (first few knots):')
print(f'  {"j":<4} {"tau":<10} {"B(tau)":<12} {"PV(d>t)":<10} {"B_ex":<10} {"S_ex/B_ex":<10} {"d1":<10} {"integrand f":<15}')

f_vec_div = np.zeros(N)
for j in range(min(10, N)):
    tau_j = 1.0 - y_div[j]**2
    if tau_j < 1e-12:
        continue
    
    B_j_dirty = B_dirty[j]
    pv_j = pricer_with_div._pv_divs(tau_j)  # PV of divs from t_j to T
    B_ex_j = B_j_dirty - pv_j
    
    vt = sigma * np.sqrt(tau_j)
    d1 = (np.log(S_ex / B_ex_j) + (r - q + 0.5 * sigma**2) * tau_j) / vt
    d2 = d1 - vt
    
    # Call option (w=+1)
    f_j = 2 * y_div[j] * (
        q * S_ex * np.exp(-q * tau_j) * norm.cdf(d1)
        - r * 100 * np.exp(-r * tau_j) * norm.cdf(d2)
    )
    f_vec_div[j] = f_j
    
    print(f'  {j:<4} {tau_j:<10.6f} {B_j_dirty:<12.4f} {pv_j:<10.4f} {B_ex_j:<10.4f} {S_ex/B_ex_j:<10.6f} {d1:<10.4f} {f_j:<15.8f}')

print()
print('=' * 80)
print('COMPARISON')
print('=' * 80)
print(f'Delta EEP: {res_with_div["eep"] - res_no_div["eep"]:+.8f}')
print()
print('For a CALL, S_ex/B_ex should be LARGER with dividends (boundary drops more)')
print('This should make d1 larger and f_j larger, increasing EEP')
