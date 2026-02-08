import numpy as np
from kim_integral_rbf import KimIntegralRBF

pricer = KimIntegralRBF(K=100, T=1.0, r=0.05, q=0.05, sigma=0.25, w=+1, N=60, n_powers=1,
                        div_times=[0.4], div_amounts=[10.0])
res = pricer.price(S0=100, max_iters=10, verbose_diagnostics=False)

# Get residuals
details = pricer._last_residual_details
B_final = details['B']
euro_final = details['euro_p']
eep_final = details['eep']
residual_full = pricer.w * (B_final - pricer.K) - (euro_final + eep_final)

tau_d = 1.0 - 0.4
idx_div = np.argmin(np.abs(details['tau'] - tau_d))

print(f'tau_d = {tau_d}')
print(f'idx_div = {idx_div}')
print(f'tau[{idx_div}] = {details["tau"][idx_div]}')
print(f'tau[{idx_div+1}] = {details["tau"][idx_div+1]}')
print(f'residual_full[{idx_div}] = {residual_full[idx_div]}')
print(f'residual_full[{idx_div+1}] = {residual_full[idx_div+1]}')
print(f'abs(residual_full[{idx_div}]) = {abs(residual_full[idx_div])}')
print(f'abs(residual_full[{idx_div+1}]) = {abs(residual_full[idx_div+1])}')
print(f'jump = {abs(residual_full[idx_div+1]) / abs(residual_full[idx_div])}')
