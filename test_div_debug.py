import numpy as np
from new_american import AmericanRBFDividendEngine

S0, K, T, r, q, sigma, w = 100, 100, 1.0, 0.05, 0.0, 0.25, 1
dividends = [(0.25, 2.0), (0.75, 2.0)]

pricer = AmericanRBFDividendEngine(K=K, T=T, r=r, q=q, sigma=sigma, 
                                   dividends=dividends, w=w, n_knots=21)

H = pricer._get_initial_boundary_with_divs()
print(f"Initial H: {H}")
print(f"Initial B: {np.exp(H)}")

try:
    res, jac = pricer.get_residual_and_jac(H)
    print(f"\nResidual shape: {res.shape}")
    print(f"Jacobian shape: {jac.shape}")
    print(f"Jacobian condition number: {np.linalg.cond(jac)}")
    print(f"\nJacobian rank: {np.linalg.matrix_rank(jac)}")
    print(f"Expected rank: {len(H)}")
    
    # Check for NaNs or Infs
    print(f"\nNaNs in Jacobian: {np.any(np.isnan(jac))}")
    print(f"Infs in Jacobian: {np.any(np.isinf(jac))}")
    
    # Print diagonal
    print(f"\nJacobian diagonal: {np.diag(jac)}")
    
except Exception as e:
    print(f"Error computing Jacobian: {e}")
    import traceback
    traceback.print_exc()
