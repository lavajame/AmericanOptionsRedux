
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from american_options.ju_1998 import Ju1998Pricing
from american_options.fd_pricer import FiniteDifferenceAmerican
from american_options.kim_jang_iter import KimJangIterative
from american_options.gauss_kronrod import GKRule

FDM_M = 200
FDM_N = 200
FDM_MAX_ITER = 200

def run_verification():
    results = []
    
    # --- Table I: American Calls (S, sigma, r, delta) ---
    # K=100, T=0.5
    # Columns: S, sigma, r, delta, EXP3_Paper
    data_tab1 = [
        (80, 0.2, 0.03, 0.07, 0.2196),
        (90, 0.2, 0.03, 0.07, 1.3872),
        (100, 0.2, 0.03, 0.07, 4.7837),
        (110, 0.2, 0.03, 0.07, 11.0993),
        (120, 0.2, 0.03, 0.07, 20.0005),
        
        # Vol 0.4
        (80, 0.4, 0.03, 0.07, 2.6899),
        (90, 0.4, 0.03, 0.07, 5.7237),
        (100, 0.4, 0.03, 0.07, 10.2404),
        (110, 0.4, 0.03, 0.07, 16.1831),
        (120, 0.4, 0.03, 0.07, 23.3622),

        # Vol 0.3, r=0.00, d=0.07
        (80, 0.3, 0.00, 0.07, 1.0381),
        (90, 0.3, 0.00, 0.07, 3.1247),
        (100, 0.3, 0.00, 0.07, 7.0371),
        (110, 0.3, 0.00, 0.07, 12.9574),
        (120, 0.3, 0.00, 0.07, 20.7194),
    ]
    
    K_ref = 100.0
    T_ref = 0.5
    
    print("--- Table 1: American Calls (K=100, T=0.5) ---")
    print(f"{'S':<5} {'Vol':<5} {'r':<5} {'d':<5} {'Paper':<8} {'Ju':<8} {'FDM':<8} {'KJ':<8} {'JuDiff':<8} {'FDMDiff':<8} {'KJDiff':<8}")
    
    for S, sigma, r, delta, ref_val in data_tab1:
        ju_price = Ju1998Pricing(S, K_ref, r, T_ref, sigma, delta, "call").price()

        fdm_price = FiniteDifferenceAmerican(
            S,
            K_ref,
            r,
            T_ref,
            sigma,
            q=delta,
            option_type="call",
            is_american=True,
            M=FDM_M,
            N=FDM_N,
            omega=1.2,
            tol=1e-10,
            max_iter=FDM_MAX_ITER,
            rannacher_steps=2,
        ).price()

        kj_price = KimJangIterative(
            K=K_ref,
            r=r,
            T=T_ref,
            sigma=sigma,
            alpha=delta,
            option_type="call",
            gk_rule=GKRule.ULTRA,
        ).price(S, T_ref)

        ju_diff = ju_price - ref_val
        fdm_diff = fdm_price - ref_val
        kj_diff = kj_price - ref_val

        results.append({
            'Type': 'Call',
            'S': S,
            'Vol': sigma,
            'r': r,
            'd': delta,
            'Ref': ref_val,
            'Ju': ju_price,
            'FDM': fdm_price,
            'KJ': kj_price,
            'JuDiff': ju_diff,
            'FDMDiff': fdm_diff,
            'KJDiff': kj_diff,
        })
        print(f"{S:<5} {sigma:<5} {r:<5} {delta:<5} {ref_val:<8.4f} {ju_price:<8.4f} {fdm_price:<8.4f} {kj_price:<8.4f} {ju_diff:<8.4f} {fdm_diff:<8.4f} {kj_diff:<8.4f}")

    print("\n--- Exhibit 3 (Ju 99, but checking EXP3 logic): American Puts (S=40, r=0.0488, d=0) ---")
    # For Puts, check if numbers align with Ju 98 methods generally or just robustness
    # (35, 0.2, 0.0833) -> Paper 0.006 (Mquad). EXP3 should be close.
    data_ex3 = [
        (35, 0.2, 0.0833, 0.006),
        (40, 0.2, 0.3333, 1.576), # Mquad value as proxy
    ]
    S_ref_put = 40.0
    r_put = 0.0488
    d_put = 0.0
    
    print(f"{'K':<5} {'Vol':<5} {'T':<7} {'Paper':<8} {'Ju':<8} {'FDM':<8} {'KJ':<8} {'JuDiff':<8} {'FDMDiff':<8} {'KJDiff':<8}")
    for K, sigma, T, ref_val in data_ex3:
        ju_price = Ju1998Pricing(S_ref_put, K, r_put, T, sigma, d_put, "put").price()

        fdm_price = FiniteDifferenceAmerican(
            S_ref_put,
            K,
            r_put,
            T,
            sigma,
            q=d_put,
            option_type="put",
            is_american=True,
            M=FDM_M,
            N=FDM_N,
            omega=1.2,
            tol=1e-10,
            max_iter=FDM_MAX_ITER,
            rannacher_steps=2,
        ).price()

        kj_price = KimJangIterative(
            K=K,
            r=r_put,
            T=T,
            sigma=sigma,
            alpha=d_put,
            option_type="put",
            gk_rule=GKRule.ULTRA,
        ).price(S_ref_put, T)

        ju_diff = ju_price - ref_val
        fdm_diff = fdm_price - ref_val
        kj_diff = kj_price - ref_val

        print(f"{K:<5} {sigma:<5} {T:<7} {ref_val:<8.4f} {ju_price:<8.4f} {fdm_price:<8.4f} {kj_price:<8.4f} {ju_diff:<8.4f} {fdm_diff:<8.4f} {kj_diff:<8.4f}")

    df = pd.DataFrame(results)
    for label in ["Ju", "FDM", "KJ"]:
        mae = df[f"{label}Diff"].abs().mean()
        rmse = np.sqrt((df[f"{label}Diff"] ** 2).mean())
        print(f"\nOverall Call {label} MAE: {mae:.4f}")
        print(f"Overall Call {label} RMSE: {rmse:.4f}")

if __name__ == "__main__":
    run_verification()
