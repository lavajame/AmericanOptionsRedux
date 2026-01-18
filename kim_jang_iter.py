import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from gauss_kronrod import GaussKronrodRule, GKRule


class KimJangIterative:
    """
    Iterative optimal exercise boundary method from:
    "A Simple Iterative Method for the Valuation of American Options" (2007).

    This implementation includes the Section V extension to continuous proportional
    dividends (alpha). The singular 1/sqrt(τ-ξ) integrands are handled via the
    substitution s = τ-ξ, s = u^2, so ds = 2u du and 1/sqrt(s) cancels.
    """

    def __init__(
        self,
        K,
        r,
        T,
        sigma,
        alpha=0.0,
        option_type="put",
        gk_rule=GKRule.FAST,
    ):
        self.K = float(K)
        self.r = float(r)
        self.T = float(T)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.option_type = option_type.lower()
        self.gk_rule = gk_rule

        if self.option_type not in {"put", "call"}:
            raise ValueError("option_type must be 'put' or 'call'")

        self._phi = 1.0 if self.option_type == "call" else -1.0

        self._gk = GaussKronrodRule(rule=self.gk_rule)

    def _d1(self, S, tau, B, r=None, alpha=None):
        if tau <= 0:
            return np.inf if S > B else -np.inf
        if not np.isfinite(B):
            return -np.inf
        if B <= 0:
            return np.inf
        if S <= 0:
            return -np.inf
        r = self.r if r is None else r
        alpha = self.alpha if alpha is None else alpha
        return (np.log(S / B) + (r - alpha + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))

    def _d2(self, S, tau, B, r=None, alpha=None):
        return self._d1(S, tau, B, r=r, alpha=alpha) - self.sigma * np.sqrt(tau)

    def _euro_put(self, S, tau, r=None, alpha=None):
        if tau <= 0:
            return max(self.K - S, 0.0)
        r = self.r if r is None else r
        alpha = self.alpha if alpha is None else alpha
        d1 = self._d1(S, tau, self.K, r=r, alpha=alpha)
        d2 = d1 - self.sigma * np.sqrt(tau)
        return self.K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-alpha * tau) * norm.cdf(-d1)

    def _euro_call(self, S, tau, r=None, alpha=None):
        if tau <= 0:
            return max(S - self.K, 0.0)
        r = self.r if r is None else r
        alpha = self.alpha if alpha is None else alpha
        d1 = self._d1(S, tau, self.K, r=r, alpha=alpha)
        d2 = d1 - self.sigma * np.sqrt(tau)
        return S * np.exp(-alpha * tau) * norm.cdf(d1) - self.K * np.exp(-r * tau) * norm.cdf(d2)

    def _boundary_initial(self):
        if self.alpha > 0:
            return min(self.K, (self.r / self.alpha) * self.K)
        return self.K

    def _call_boundary_initial(self):
        if self.alpha > 0:
            return self.K
        return np.inf

    def _call_upper_bound(self):
        if self.r <= 0 or self.sigma <= 0:
            return self.K * 20.0

        b = 0.5 - (self.r - self.alpha) / (self.sigma**2)
        disc = b * b + 2.0 * self.r / (self.sigma**2)
        beta = b + np.sqrt(disc)
        if beta <= 1.0:
            return self.K * 20.0

        return self.K * beta / (beta - 1.0)

    def _interp_boundary(self, tau_grid, B_grid):
        def B_of_tau(tau):
            if tau <= 0:
                return B_grid[0]
            if tau >= tau_grid[-1]:
                return B_grid[-1]
            return np.interp(tau, tau_grid, B_grid)

        return B_of_tau

    def _sanitize_boundary(self, B_grid, min_val=None, max_val=None, monotonic=None, max_jump_ratio=None):
        B = np.array(B_grid, dtype=float)
        B[~np.isfinite(B)] = np.nan

        if np.all(np.isnan(B)):
            return B_grid

        idx = np.arange(len(B))
        finite = np.isfinite(B)
        B = np.interp(idx, idx[finite], B[finite])

        if min_val is not None:
            B = np.maximum(B, min_val)
        if max_val is not None:
            B = np.minimum(B, max_val)

        if monotonic == "nondecreasing":
            B = np.maximum.accumulate(B)
            if max_jump_ratio is not None:
                for i in range(1, len(B)):
                    if B[i] > B[i - 1] * max_jump_ratio:
                        B[i] = B[i - 1] * max_jump_ratio
        elif monotonic == "nonincreasing":
            B = np.minimum.accumulate(B)

        return B

    def _gk_integrate(self, func, a, b):
        return self._gk.integrate(func, a, b)

    def _integral_singular(self, tau, B_tau, B_prev_func, use_d1=False):
        # Integral of exp(-r s - 0.5 d^2)/(sigma*sqrt(2π s)) ds from s=0..tau
        if tau <= 0:
            return 0.0

        def integrand_u(u):
            s = u * u
            B_xi = B_prev_func(max(tau - s, 0.0))
            if use_d1:
                d = self._d1(B_tau, s, B_xi)
            else:
                d = self._d2(B_tau, s, B_xi)
            return 2.0 * np.exp(-self.r * s - 0.5 * d * d) / (self.sigma * np.sqrt(2 * np.pi))

        upper = np.sqrt(tau)
        return self._gk_integrate(integrand_u, 0.0, upper)

    def _integral_nonsingular(self, tau, B_tau, B_prev_func):
        # Integral of exp(-r s) * N(d1) ds from s=0..tau
        if tau <= 0:
            return 0.0

        def integrand_s(s):
            B_xi = B_prev_func(max(tau - s, 0.0))
            d1 = self._d1(B_tau, s, B_xi)
            return np.exp(-self.r * s) * norm.cdf(self._phi * d1)

        return self._gk_integrate(integrand_s, 0.0, tau)

    def _boundary_update(self, tau, B_tau_prev, B_prev_func):
        if tau <= 0:
            return self._boundary_initial()

        d1_tau = self._d1(B_tau_prev, tau, self.K)
        d2_tau = d1_tau - self.sigma * np.sqrt(tau)

        term0 = (self.K / (self.sigma * np.sqrt(2 * np.pi * tau))) * np.exp(-self.r * tau - 0.5 * d2_tau * d2_tau)
        I2 = self._integral_singular(tau, B_tau_prev, B_prev_func, use_d1=False)
        numerator = term0 + self.r * self.K * I2

        denom0 = norm.cdf(self._phi * d1_tau) + (1.0 / (self.sigma * np.sqrt(2 * np.pi * tau))) * np.exp(-0.5 * d1_tau * d1_tau)

        if self.alpha <= 0:
            return numerator / denom0

        denom_div = np.exp(-self.alpha * tau) * norm.cdf(self._phi * d1_tau)
        denom_div += (1.0 / (self.sigma * np.sqrt(2 * np.pi * tau))) * np.exp(-self.alpha * tau - 0.5 * d1_tau * d1_tau)

        I1 = self._integral_nonsingular(tau, B_tau_prev, B_prev_func)
        I1_pdf = self._integral_singular(tau, B_tau_prev, B_prev_func, use_d1=True)
        denom_div += self.alpha * (I1 + I1_pdf)

        return numerator / denom_div

    def compute_boundary(self, n_nodes=20, n_iter=8):
        # tau grid (time to maturity)
        tau_grid = np.linspace(0.0, self.T, n_nodes)

        if self.option_type == "call":
            return self._compute_call_boundary_dual(tau_grid)

        # Initial guess B0(τ)
        B0 = np.full_like(tau_grid, self._boundary_initial(), dtype=float)
        if n_iter <= 0:
            return tau_grid, B0

        # First iteration: use B0 on RHS (explicit, Eq. 6 analogue with dividends)
        B_prev_func = self._interp_boundary(tau_grid, B0)
        B1 = B0.copy()
        for i, tau in enumerate(tau_grid):
            if tau <= 0:
                B1[i] = self._boundary_initial()
            else:
                B1[i] = self._boundary_update(tau, B_prev_func(tau), B_prev_func)

        B_prev = self._sanitize_boundary(B1, min_val=0.0, max_val=self.K, monotonic="nonincreasing")

        for _ in range(n_iter - 1):
            B_prev_func = self._interp_boundary(tau_grid, B_prev)
            B_new = B_prev.copy()
            for i, tau in enumerate(tau_grid):
                if tau <= 0:
                    B_new[i] = self._boundary_initial()
                else:
                    B_new[i] = self._boundary_update(tau, B_prev_func(tau), B_prev_func)
            B_prev = self._sanitize_boundary(B_new, min_val=0.0, max_val=self.K, monotonic="nonincreasing")

        return tau_grid, B_prev

    def _compute_call_boundary(self, tau_grid, n_iter=8, damping=0.6):
        # For calls with continuous dividends, solve C_cont(B,tau) = B - K.
        # If alpha <= 0, early exercise is not optimal: boundary is infinite.
        if self.alpha <= 0:
            return tau_grid, np.full_like(tau_grid, np.inf, dtype=float)

        # Initial guess
        B_prev = np.full_like(tau_grid, self._call_boundary_initial(), dtype=float)
        if n_iter <= 0:
            B_prev = self._sanitize_boundary(B_prev, min_val=self.K, monotonic="nondecreasing")
            return tau_grid, B_prev

        upper_bound = max(self._call_upper_bound(), self.K * 50.0)

        for _ in range(n_iter):
            B_prev_func = self._interp_boundary(tau_grid, B_prev)
            B_new = B_prev.copy()

            for i, tau in enumerate(tau_grid):
                if tau <= 0:
                    B_new[i] = self.K
                    continue

                def f(B):
                    cont = self._continuation_call_value(B, tau, B_prev_func)
                    return cont - (B - self.K)

                low = self.K
                try:
                    f_low = f(low)
                except Exception:
                    f_low = np.nan

                if not np.isfinite(f_low):
                    B_new[i] = np.nan
                    continue

                if f_low < 0:
                    B_new[i] = self.K
                    continue

                # Search for a sign change on a geometric grid
                grid = np.geomspace(self.K, upper_bound, 24)
                f_vals = []
                for g in grid:
                    try:
                        f_vals.append(f(g))
                    except Exception:
                        f_vals.append(np.nan)

                bracket = None
                for j in range(1, len(grid)):
                    f0, f1 = f_vals[j - 1], f_vals[j]
                    if np.isfinite(f0) and np.isfinite(f1) and f0 * f1 <= 0:
                        bracket = (grid[j - 1], grid[j])
                        break

                if bracket is None:
                    # No sign change found: keep previous value to avoid biasing low
                    B_new[i] = B_prev[i] if np.isfinite(B_prev[i]) else np.nan
                    continue

                try:
                    low, high = bracket
                    root = brentq(f, low, high, maxiter=200)
                    if np.isfinite(B_prev[i]):
                        B_new[i] = (1.0 - damping) * B_prev[i] + damping * root
                    else:
                        B_new[i] = root
                except Exception:
                    B_new[i] = np.nan

                if np.isfinite(B_new[i]) and B_new[i] < self.K:
                    B_new[i] = self.K

            B_prev = self._sanitize_boundary(
                B_new,
                min_val=self.K,
                max_val=upper_bound,
                monotonic="nondecreasing",
                max_jump_ratio=1.2,
            )

        return tau_grid, B_prev

    def _dual_put_price(self, strike, tau):
        dual = KimJangIterative(
            K=strike,
            r=self.alpha,
            T=self.T,
            sigma=self.sigma,
            alpha=self.r,
            option_type="put",
            gk_rule=self.gk_rule,
        )
        return dual.price_put(self.K, tau)

    def _compute_call_boundary_dual(self, tau_grid):
        if self.alpha <= 0:
            return tau_grid, np.full_like(tau_grid, np.inf, dtype=float)

        upper_bound = max(self._call_upper_bound(), self.K * 50.0)
        upper_bound = max(upper_bound, self.K * 1.5)

        B_vals = np.full_like(tau_grid, np.nan, dtype=float)

        for i, tau in enumerate(tau_grid):
            if tau <= 0:
                B_vals[i] = self.K
                continue

            def f(B):
                return self._dual_put_price(B, tau) - (B - self.K)

            grid = np.geomspace(self.K, upper_bound, 28)
            f_vals = []
            for g in grid:
                try:
                    f_vals.append(f(g))
                except Exception:
                    f_vals.append(np.nan)

            bracket = None
            for j in range(1, len(grid)):
                f0, f1 = f_vals[j - 1], f_vals[j]
                if np.isfinite(f0) and np.isfinite(f1) and f0 * f1 <= 0:
                    bracket = (grid[j - 1], grid[j])
                    break

            if bracket is None:
                B_vals[i] = np.nan
                continue

            try:
                B_vals[i] = brentq(f, bracket[0], bracket[1], maxiter=200)
            except Exception:
                B_vals[i] = np.nan

        B_vals = self._sanitize_boundary(
            B_vals,
            min_val=self.K,
            max_val=upper_bound,
            monotonic="nondecreasing",
            max_jump_ratio=1.2,
        )

        return tau_grid, B_vals

    def price_put(self, S, tau, boundary=None):
        if tau <= 0:
            return max(self.K - S, 0.0)

        if boundary is None:
            _, B_grid = self.compute_boundary()
            tau_grid = np.linspace(0.0, self.T, len(B_grid))
            B_func = self._interp_boundary(tau_grid, B_grid)
        else:
            tau_grid, B_grid = boundary
            B_func = self._interp_boundary(tau_grid, B_grid)

        B_tau = B_func(tau)
        if S <= B_tau:
            return self.K - S

        return self._continuation_put_value(S, tau, B_func)

    def _continuation_put_value(self, S, tau, B_func):
        euro = self._euro_put(S, tau)

        def integrand_s(s):
            B_xi = B_func(max(tau - s, 0.0))
            d2 = self._d2(S, s, B_xi)
            term = self.r * self.K * np.exp(-self.r * s) * norm.cdf(-d2)
            if self.alpha > 0:
                d1 = self._d1(S, s, B_xi)
                term -= self.alpha * S * np.exp(-self.alpha * s) * norm.cdf(-d1)
            return term

        prem = self._gk_integrate(integrand_s, 0.0, tau)
        return euro + prem

    def _continuation_call_value(self, S, tau, B_func):
        euro = self._euro_call(S, tau)

        def integrand_s(s):
            B_xi = B_func(max(tau - s, 0.0))
            d1 = self._d1(S, s, B_xi)
            d2 = d1 - self.sigma * np.sqrt(s)
            term = self.alpha * S * np.exp(-self.alpha * s) * norm.cdf(d1)
            term -= self.r * self.K * np.exp(-self.r * s) * norm.cdf(d2)
            return term

        prem = self._gk_integrate(integrand_s, 0.0, tau)
        return euro + prem

    def price(self, S, tau, boundary=None):
        if self.option_type == "put":
            return self.price_put(S, tau, boundary=boundary)

        return self.price_call(S, tau, boundary=boundary)

    def price_call(self, S, tau, boundary=None):
        if tau <= 0:
            return max(S - self.K, 0.0)

        if self.alpha <= 0:
            return self._euro_call(S, tau)

        # Put-call duality for American options with continuous dividends
        return self._dual_put_price(S, tau)


if __name__ == "__main__":
    # Basic smoke test
    pricer = KimJangIterative(K=100, r=0.05, T=0.5, sigma=0.2, alpha=0.0, option_type="put")
    tau_grid, B = pricer.compute_boundary(n_nodes=10, n_iter=4)
    print("Boundary (tau, B):", list(zip(tau_grid, B))[:5])
    print("Price:", pricer.price(100, 0.5))
