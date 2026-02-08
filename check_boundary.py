import numpy as np
import pandas as pd
from new_american import AmericanRBFDividendEngine

S0, K, T, r, q, sigma, w = 100, 100, 1.0, 0.05, 0.0, 0.25, -1  # PUT
dividends = [(0.25, 3.0), (0.75, 3.0)]

pricer = AmericanRBFDividendEngine(K=K, T=T, r=r, q=q, sigma=sigma, 
                                   dividends=dividends, w=w, n_knots=31)

results = pricer.price(S0=S0, max_iters=10)

# Show full boundary
df = pd.DataFrame({
    'tau': pricer.y**2,
    't': T - pricer.y**2,
    'boundary': results['boundary']['B'].values
})

print("Full Boundary:")
print(df.to_string(index=False))
print(f"\nMax: {df['boundary'].max():.2f}")
print(f"Min: {df['boundary'].min():.2f}")

# Check near dividend dates
for t_div, d in dividends:
    nearby = df[np.abs(df['t'] - t_div) < 0.05]
    print(f"\nNear dividend at t={t_div}:")
    print(nearby.to_string(index=False))
