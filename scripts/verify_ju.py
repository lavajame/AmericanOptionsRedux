
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from american_options.ju_1998 import pricing_function, Ju1998Pricing

def run_verification():
    results = []
    
    # --- Exhibit 3: American Puts ---
    # S=40, r=0.0488, delta=0.0
    # Columns: K, sigma, T, Mquad_Paper
    data_ex3 = [
        (35, 0.2, 0.0833, 0.006),
        (35, 0.2, 0.3333, 0.201),
        (35, 0.2, 0.5833, 0.433),
        (40, 0.2, 0.0833, 0.851),
        (40, 0.2, 0.3333, 1.576),
        (40, 0.2, 0.5833, 1.984),
        (45, 0.2, 0.0833, 5.000),
        (45, 0.2, 0.3333, 5.084),
        (45, 0.2, 0.5833, 5.260),
        (40, 0.3, 0.0833, 1.309), # ATM
        (40, 0.3, 0.3333, 2.477),
        (40, 0.3, 0.5833, 3.161),
        (40, 0.4, 0.0833, 1.767), # ATM High Vol
        (40, 0.4, 0.3333, 3.381),
        (40, 0.4, 0.5833, 4.342)
    ]
    
    S_ref = 40.0
    r_ref = 0.0488
    delta_ref = 0.0
    
    print("--- Exhibit 3: American Puts (S=40, r=0.0488, d=0) ---")
    print(f"{'K':<5} {'Vol':<5} {'T':<8} {'Paper':<8} {'Calcd':<8} {'Diff':<8}")
    
    for K, sigma, T, ref_val in data_ex3:
        price = pricing_function(S_ref, K, r_ref, T, sigma, delta_ref, 'put')
        diff = price - ref_val
        results.append({'Exhibit': '3', 'Type': 'Put', 'K': K, 'T': T, 'Sigma': sigma, 'Ref': ref_val, 'Calc': price, 'Diff': diff})
        print(f"{K:<5} {sigma:<5} {T:<8.4f} {ref_val:<8.3f} {price:<8.3f} {diff:<8.4f}")

    # --- Exhibit 4: American Calls ---
    # K=100, T=0.5
    # Columns: S, sigma, r, delta, Mquad_Paper
    data_ex4 = [
        (80, 0.2, 0.03, 0.07, 0.222),
        (90, 0.2, 0.03, 0.07, 1.386),
        (100, 0.2, 0.03, 0.07, 4.768),
        (110, 0.2, 0.03, 0.07, 11.079),
        (120, 0.2, 0.03, 0.07, 20.000),
        (100, 0.4, 0.03, 0.07, 10.214), # ATM High Vol
        (100, 0.3, 0.00, 0.07, 7.015), # ATM Zero rate? No r=0.00 in table?
        # Check Exhibit 4: r=0.00, delta=0.07 in table?
        # Row 11: (80, 0.3, 0.00, 0.07) -> Mquad 1.040
        # Row 13: (100, 0.3, 0.00, 0.07) -> Mquad 7.015
    ]
    
    K_ref = 100.0
    T_ref = 0.5
    
    print("\n--- Exhibit 4: American Calls (K=100, T=0.5) ---")
    print(f"{'S':<5} {'Vol':<5} {'r':<5} {'d':<5} {'Paper':<8} {'Calcd':<8} {'Diff':<8}")
    
    for S, sigma, r, delta, ref_val in data_ex4:
        price = pricing_function(S, K_ref, r, T_ref, sigma, delta, 'call')
        diff = price - ref_val
        results.append({'Exhibit': '4', 'Type': 'Call', 'S': S, 'r': r, 'd': delta, 'Ref': ref_val, 'Calc': price, 'Diff': diff})
        print(f"{S:<5} {sigma:<5} {r:<5} {delta:<5} {ref_val:<8.3f} {price:<8.3f} {diff:<8.4f}")

    df = pd.DataFrame(results)
    mae = df['Diff'].abs().mean()
    rmse = np.sqrt((df['Diff']**2).mean())
    print(f"\nOverall MAE: {mae:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")

if __name__ == "__main__":
    run_verification()
