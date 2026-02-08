"""
Test script to compare EEP with and without discrete dividends
"""
import numpy as np
from kim_integral_rbf import KimIntegralRBF

# Test parameters
N = 20
max_iters = 5
n_powers = 3
r = 0.05
q = 0.05
sigma = 0.25

print('=' * 60)
print('CASE 1: NO DIVIDENDS')
print('=' * 60)
pricer_no_div = KimIntegralRBF(
    K=100, T=1.0, r=r, q=q, sigma=sigma, w=+1, N=N, n_powers=n_powers,
    div_times=[], div_amounts=[])
res_no_div = pricer_no_div.price(S0=100, max_iters=max_iters, verbose_diagnostics=False)
print(f'Euro : {res_no_div["euro"]:.8f}')
print(f'EEP  : {res_no_div["eep"]:.8f}')
print(f'Amer : {res_no_div["amer"]:.8f}')
print()

print('=' * 60)
print('CASE 2: WITH DISCRETE DIVIDEND D=5.0 at t=0.4')
print('=' * 60)
pricer_with_div = KimIntegralRBF(
    K=100, T=1.0, r=r, q=q, sigma=sigma, w=+1, N=N, n_powers=n_powers,
    div_times=[0.4], div_amounts=[5.0])
res_with_div = pricer_with_div.price(S0=100, max_iters=max_iters, verbose_diagnostics=False)
print(f'Euro : {res_with_div["euro"]:.8f}')
print(f'EEP  : {res_with_div["eep"]:.8f}')
print(f'Amer : {res_with_div["amer"]:.8f}')
print()

print('=' * 60)
print('COMPARISON')
print('=' * 60)
print(f'Delta Euro : {res_with_div["euro"] - res_no_div["euro"]:+.8f}')
print(f'Delta EEP  : {res_with_div["eep"] - res_no_div["eep"]:+.8f}')
print(f'Delta Amer : {res_with_div["amer"] - res_no_div["amer"]:+.8f}')
print()
print('For a CALL, discrete dividends should INCREASE EEP')
print('(want to exercise before dividend to avoid drop in stock price)')
if res_with_div["eep"] < res_no_div["eep"]:
    print('>>> PROBLEM: EEP is LOWER with dividends! <<<')
else:
    print('OK: EEP is higher with dividends as expected')
