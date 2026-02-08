"""
Final verification: EEP should increase with discrete dividends for calls
"""
import numpy as np
from kim_integral_rbf import KimIntegralRBF

params = {
    'K': 100, 'T': 1.0, 'r': 0.05, 'q': 0.05, 'sigma': 0.25,
    'N': 20, 'max_iters': 5, 'n_powers': 3
}

print('=' * 70)
print('VERIFICATION: EEP behavior with discrete dividends')
print('=' * 70)
print(f'Params: K={params["K"]}, T={params["T"]}, r={params["r"]}, q={params["q"]}, σ={params["sigma"]}')
print()

cases = [
    ('No dividends', [], []),
    ('D=2.0 at t=0.4', [0.4], [2.0]),
    ('D=5.0 at t=0.4', [0.4], [5.0]),
    ('D=10.0 at t=0.4', [0.4], [10.0]),
]

results = []
for name, div_t, div_a in cases:
    pricer = KimIntegralRBF(
        K=params['K'], T=params['T'], r=params['r'], q=params['q'], 
        sigma=params['sigma'], w=+1, N=params['N'], n_powers=params['n_powers'],
        div_times=div_t, div_amounts=div_a)
    res = pricer.price(S0=100, max_iters=params['max_iters'], verbose_diagnostics=False)
    results.append((name, res))
    
    pv = pricer._pv_divs(params['T'])
    print(f'{name:20s} | PV(D)={pv:6.2f} | Euro={res["euro"]:8.4f} | EEP={res["eep"]:8.4f} | Amer={res["amer"]:8.4f}')

print()
print('=' * 70)
print('ANALYSIS')
print('=' * 70)

base_eep = results[0][1]['eep']
print(f'Base EEP (no divs): {base_eep:.6f}\n')

for i in range(1, len(results)):
    name, res = results[i]
    delta_eep = res['eep'] - base_eep
    pct_change = (delta_eep / base_eep) * 100
    status = '✓' if delta_eep > 0 else '✗'
    print(f'{status} {name:20s}: EEP = {res["eep"]:.6f}  (Δ = {delta_eep:+.6f}, {pct_change:+.1f}%)')

print()
print('For CALL options, discrete dividends should INCREASE EEP.')
print('Reason: Incentive to exercise before dividend to avoid stock price drop.')
