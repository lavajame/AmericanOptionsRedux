from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from typing import Callable, Iterable, List, Tuple, cast

import numpy as np

from gauss_kronrod import GaussKronrodRule, GKRule


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


@dataclass
class AmericanPutBoundaryGK:
    """
    Early-exercise boundary for an American put using a fixed-point iteration
    with Gauss–Kronrod quadrature for the required integrals.

    Parameters
    ----------
    K : float
        Strike.
    r : float
        Risk-free rate.
    q : float
        Continuous dividend yield.
    sigma : float
        Volatility.
    T : float
        Maturity (time to expiry).
    gk_rule : GKRule
        Fixed Gauss–Kronrod rule (default BALANCED).
    """

    K: float
    r: float
    q: float
    sigma: float
    T: float
    gk_rule: GKRule = GKRule.BALANCED

    def __post_init__(self) -> None:
        self.K = float(self.K)
        self.r = float(self.r)
        self.q = float(self.q)
        self.sigma = float(self.sigma)
        self.T = float(self.T)
        self._gk = GaussKronrodRule(rule=self.gk_rule)

    def d1(self, S: float, B: float, tau: float) -> float:
        if tau <= 0.0:
            return float("inf") if S > B else float("-inf")
        if S <= 0.0 or B <= 0.0:
            return float("-inf")
        return (log(S / B) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / (
            self.sigma * sqrt(tau)
        )

    def d2(self, S: float, B: float, tau: float) -> float:
        return self.d1(S, B, tau) - self.sigma * sqrt(tau)

    def _boundary_initial(self) -> float:
        if self.q > 0.0:
            return min(self.K, (self.r / self.q) * self.K)
        return self.K

    def _interp_boundary(self, tau_grid: np.ndarray, B_grid: np.ndarray) -> Callable[[float], float]:
        def B_of_tau(tau: float) -> float:
            if tau <= tau_grid[0]:
                return float(B_grid[0])
            if tau >= tau_grid[-1]:
                return float(B_grid[-1])
            return float(np.interp(tau, tau_grid, B_grid))

        return B_of_tau

    def _sanitize_boundary(self, B_grid: np.ndarray) -> np.ndarray:
        B = np.array(B_grid, dtype=float)
        B[~np.isfinite(B)] = np.nan

        if np.all(np.isnan(B)):
            return B_grid

        idx = np.arange(len(B))
        finite = np.isfinite(B)
        B = np.interp(idx, idx[finite], B[finite])

        B = np.minimum(self.K, np.maximum(0.0, B))
        B = np.minimum.accumulate(B)
        return B

    def _sanitize_call_boundary(self, S_grid: np.ndarray) -> np.ndarray:
        S = np.array(S_grid, dtype=float)
        S[~np.isfinite(S)] = np.nan

        if np.all(np.isnan(S)):
            return S_grid

        idx = np.arange(len(S))
        finite = np.isfinite(S)
        S = np.interp(idx, idx[finite], S[finite])

        S = np.maximum(self.K, S)
        S = np.maximum.accumulate(S)
        return S

    def _gk_integrate(self, func: Callable[[float], float], a: float, b: float) -> float:
        return self._gk.integrate(func, a, b)

    def _integral_singular(
        self,
        tau: float,
        B_tau: float,
        B_prev_func: Callable[[float], float],
        use_d1: bool = False,
        div_schedule: Iterable[Tuple[float, float]] | None = None,
        tau_abs: float | None = None,
    ) -> float:
        """
        Integral of exp(-r s - 0.5 d^2)/(sigma*sqrt(2π s)) ds from s=0..tau
        with substitution s = u^2 (so ds = 2u du).
        """
        if tau <= 0.0:
            return 0.0

        def integrand_u(u: float) -> float:
            s = u * u
            B_xi = B_prev_func(max(tau - s, 0.0))
            if div_schedule is not None and tau_abs is not None:
                shift_tau = self._div_shift(tau_abs, div_schedule)
                shift_xi = self._div_shift(tau_abs - s, div_schedule)
                B_tau_eff = max(B_tau - shift_tau, 0.0)
                B_xi_eff = max(B_xi - shift_xi, 0.0)
                d = self.d1(B_tau_eff, B_xi_eff, s) if use_d1 else self.d2(B_tau_eff, B_xi_eff, s)
            else:
                d = self.d1(B_tau, B_xi, s) if use_d1 else self.d2(B_tau, B_xi, s)
            return 2.0 * exp(-self.r * s - 0.5 * d * d) / (self.sigma * sqrt(2.0 * pi))

        return self._gk_integrate(integrand_u, 0.0, sqrt(tau))

    def _integral_nonsingular(
        self,
        tau: float,
        B_tau: float,
        B_prev_func: Callable[[float], float],
        div_schedule: Iterable[Tuple[float, float]] | None = None,
        tau_abs: float | None = None,
    ) -> float:
        """
        Integral of exp(-r s) * N(d1) ds from s=0..tau.
        """
        if tau <= 0.0:
            return 0.0

        def integrand_s(s: float) -> float:
            B_xi = B_prev_func(max(tau - s, 0.0))
            if div_schedule is not None and tau_abs is not None:
                shift_tau = self._div_shift(tau_abs, div_schedule)
                shift_xi = self._div_shift(tau_abs - s, div_schedule)
                B_tau_eff = max(B_tau - shift_tau, 0.0)
                B_xi_eff = max(B_xi - shift_xi, 0.0)
                d1 = self.d1(B_tau_eff, B_xi_eff, s)
            else:
                d1 = self.d1(B_tau, B_xi, s)
            return exp(-self.r * s) * norm_cdf(d1)

        return self._gk_integrate(integrand_s, 0.0, tau)

    def _div_shift(self, tau_abs: float, div_schedule: Iterable[Tuple[float, float]]) -> float:
        if tau_abs <= 0.0:
            return 0.0
        t = self.T - tau_abs
        shift = 0.0
        for t_div, D in div_schedule:
            if t_div > t:
                shift += D
        return shift

    def _future_div_shift_elapsed(self, elapsed: float, div_schedule: Iterable[Tuple[float, float]]) -> float:
        if elapsed < 0.0:
            return 0.0
        shift = 0.0
        for t_div, D in div_schedule:
            if t_div > elapsed:
                shift += D
        return shift

    def _boundary_update(
        self,
        tau: float,
        B_tau_prev: float,
        B_prev_func: Callable[[float], float],
        tau_abs: float | None = None,
        div_schedule: Iterable[Tuple[float, float]] | None = None,
    ) -> float:
        if tau <= 0.0:
            return self._boundary_initial()

        tau_eff = tau if tau_abs is None else tau_abs

        if div_schedule is not None and tau_abs is not None:
            shift_tau = self._div_shift(tau_abs, div_schedule)
            B_tau_eff = max(B_tau_prev - shift_tau, 0.0)
        else:
            B_tau_eff = B_tau_prev

        d1_tau = self.d1(B_tau_eff, self.K, tau_eff)
        d2_tau = d1_tau - self.sigma * sqrt(tau_eff)

        term0 = (self.K / (self.sigma * sqrt(2.0 * pi * tau_eff))) * exp(
            -self.r * tau_eff - 0.5 * d2_tau * d2_tau
        )
        I2 = self._integral_singular(
            tau,
            B_tau_prev,
            B_prev_func,
            use_d1=False,
            div_schedule=div_schedule,
            tau_abs=tau_abs,
        )
        numerator = term0 + self.r * self.K * I2

        if self.q <= 0.0:
            denom0 = norm_cdf(d1_tau) + (1.0 / (self.sigma * sqrt(2.0 * pi * tau_eff))) * exp(
                -0.5 * d1_tau * d1_tau
            )
            return numerator / denom0

        denom_div = exp(-self.q * tau_eff) * norm_cdf(d1_tau)
        denom_div += (
            exp(-self.q * tau_eff)
            / (self.sigma * sqrt(2.0 * pi * tau_eff))
            * exp(-0.5 * d1_tau * d1_tau)
        )

        I1 = self._integral_nonsingular(
            tau,
            B_tau_prev,
            B_prev_func,
            div_schedule=div_schedule,
            tau_abs=tau_abs,
        )
        I1_pdf = self._integral_singular(
            tau,
            B_tau_prev,
            B_prev_func,
            use_d1=True,
            div_schedule=div_schedule,
            tau_abs=tau_abs,
        )
        denom_div += self.q * (I1 + I1_pdf)

        if not np.isfinite(denom_div) or denom_div <= 0.0:
            return B_tau_prev

        return numerator / denom_div

    def _call_boundary_update(
        self,
        tau: float,
        S_tau_prev: float,
        S_prev_func: Callable[[float], float],
        tau_abs: float | None = None,
        div_schedule: Iterable[Tuple[float, float]] | None = None,
    ) -> float:
        if tau <= 0.0:
            return self.K

        tau_eff = tau if tau_abs is None else tau_abs
        if div_schedule is not None and tau_abs is not None:
            shift_tau = self._div_shift(tau_abs, div_schedule)
            S_tau_eff = max(S_tau_prev - shift_tau, 0.0)
        else:
            S_tau_eff = S_tau_prev

        euro = self.european_call(S_tau_eff, tau_eff)

        def integrand_s(s: float) -> float:
            S_xi = S_prev_func(max(tau - s, 0.0))
            if div_schedule is not None and tau_abs is not None:
                shift_xi = self._div_shift(tau_abs - s, div_schedule)
                S_xi_eff = max(S_xi - shift_xi, 0.0)
                d2 = self.d2(S_tau_eff, S_xi_eff, s)
                term = -self.r * self.K * exp(-self.r * s) * norm_cdf(d2)
                d1 = self.d1(S_tau_eff, S_xi_eff, s)
                term += self.q * S_tau_eff * exp(-self.q * s) * norm_cdf(d1)
                return term
            d2 = self.d2(S_tau_eff, S_xi, s)
            term = -self.r * self.K * exp(-self.r * s) * norm_cdf(d2)
            d1 = self.d1(S_tau_eff, S_xi, s)
            term += self.q * S_tau_eff * exp(-self.q * s) * norm_cdf(d1)
            return term

        prem = self._gk_integrate(integrand_s, 0.0, tau)
        return self.K + euro + prem

    def compute_boundary(self, n_nodes: int = 30, n_iter: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the early-exercise boundary B(tau) over a tau grid.

        Returns (tau_grid, B_grid).
        """
        tau_grid = np.linspace(0.0, self.T, n_nodes)
        B0 = np.full_like(tau_grid, self._boundary_initial(), dtype=float)

        if n_iter <= 0:
            return tau_grid, B0

        B_prev_func = self._interp_boundary(tau_grid, B0)
        B_prev = B0.copy()

        for _ in range(n_iter):
            B_new = B_prev.copy()
            for i, tau in enumerate(tau_grid):
                B_new[i] = self._boundary_update(tau, B_prev_func(tau), B_prev_func)
            B_prev = self._sanitize_boundary(B_new)
            B_prev_func = self._interp_boundary(tau_grid, B_prev)

        return tau_grid, B_prev

    def compute_call_boundary(self, n_nodes: int = 30, n_iter: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the American call boundary via put-call duality and homogeneity.

        Uses the dual put boundary with swapped rates (r<->q). By homogeneity,
        the put boundary scales linearly with strike, so S*_call(tau) = K / b_dual(tau),
        where b_dual(tau) is the dual put boundary for strike=1.
        """
        tau_grid = np.linspace(0.0, self.T, n_nodes)
        if self.q <= 0.0:
            return tau_grid, np.full_like(tau_grid, np.inf, dtype=float)

        dual = AmericanPutBoundaryGK(
            K=1.0,
            r=self.q,
            q=self.r,
            sigma=self.sigma,
            T=self.T,
            gk_rule=self.gk_rule,
        )
        _, B_dual = dual.compute_boundary(n_nodes=n_nodes, n_iter=n_iter)

        with np.errstate(divide="ignore", invalid="ignore"):
            S_star = self.K / B_dual

        S_star = np.maximum(S_star, self.K)
        return tau_grid, S_star

    def compute_call_boundary_direct(self, n_nodes: int = 30, n_iter: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the American call boundary directly from the call integral equation
        using the physical r and q (no duality swap).
        """
        tau_grid = np.linspace(0.0, self.T, n_nodes)
        if self.q <= 0.0:
            return tau_grid, np.full_like(tau_grid, np.inf, dtype=float)

        S0 = np.full_like(tau_grid, self.K, dtype=float)

        if n_iter <= 0:
            return tau_grid, S0

        S_prev_func = self._interp_boundary(tau_grid, S0)
        S_prev = S0.copy()

        for _ in range(n_iter):
            S_new = S_prev.copy()
            for i, tau in enumerate(tau_grid):
                S_new[i] = self._call_boundary_update(tau, S_prev_func(tau), S_prev_func)
            S_prev = self._sanitize_call_boundary(S_new)
            S_prev_func = self._interp_boundary(tau_grid, S_prev)

        return tau_grid, S_prev

    def compute_boundary_with_terminal(
        self,
        n_nodes: int,
        n_iter: int,
        boundary_at_zero: float,
        tau_offset: float = 0.0,
        div_schedule: Iterable[Tuple[float, float]] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        tau_grid = np.linspace(0.0, self.T, n_nodes)
        B0 = np.full_like(tau_grid, boundary_at_zero, dtype=float)

        if n_iter <= 0:
            return tau_grid, B0, 0.0

        B_prev_func = self._interp_boundary(tau_grid, B0)
        B_prev = B0.copy()

        for _ in range(n_iter):
            B_new = B_prev.copy()
            for i, tau in enumerate(tau_grid):
                if i == 0:
                    B_new[i] = boundary_at_zero
                else:
                    B_new[i] = self._boundary_update(
                        tau,
                        B_prev_func(tau),
                        B_prev_func,
                        tau_abs=tau + tau_offset,
                        div_schedule=div_schedule,
                    )
            B_prev = self._sanitize_boundary(B_new)
            B_prev[0] = boundary_at_zero
            B_prev_func = self._interp_boundary(tau_grid, B_prev)

        
        boundary_abs = (tau_grid + tau_offset, B_prev)
        eep_local = self.continuation_put(
            boundary_at_zero,
            tau_offset,
            boundary=boundary_abs,
            div_schedule=div_schedule,
        ) - self.european_with_discrete_dividends("put", boundary_at_zero, tau_offset, div_schedule or [])

        return tau_grid, B_prev, float(eep_local)

    def compute_call_boundary_direct_with_terminal(
        self,
        n_nodes: int,
        n_iter: int,
        boundary_at_zero: float,
        tau_offset: float = 0.0,
        div_schedule: Iterable[Tuple[float, float]] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        tau_grid = np.linspace(0.0, self.T, n_nodes)
        S0 = np.full_like(tau_grid, boundary_at_zero, dtype=float)

        if n_iter <= 0:
            return tau_grid, S0, 0.0

        S_prev_func = self._interp_boundary(tau_grid, S0)
        S_prev = S0.copy()

        for _ in range(n_iter):
            S_new = S_prev.copy()
            for i, tau in enumerate(tau_grid):
                if i == 0:
                    S_new[i] = boundary_at_zero
                else:
                    S_new[i] = self._call_boundary_update(
                        tau,
                        S_prev_func(tau),
                        S_prev_func,
                        tau_abs=tau + tau_offset,
                        div_schedule=div_schedule,
                    )
            S_prev = self._sanitize_call_boundary(S_new)
            S_prev[0] = boundary_at_zero
            S_prev_func = self._interp_boundary(tau_grid, S_prev)

        boundary_abs = (tau_grid + tau_offset, S_prev)
        eep_local = self.continuation_call(
            boundary_at_zero,
            tau_offset,
            boundary=boundary_abs,
            div_schedule=div_schedule,
        ) - self.european_with_discrete_dividends("call", boundary_at_zero, tau_offset, div_schedule or [])

        return tau_grid, S_prev, float(eep_local)

    def compute_call_boundary_dual_with_terminal(
        self,
        n_nodes: int,
        n_iter: int,
        boundary_at_zero: float,
        tau_offset: float = 0.0,
        div_schedule: Iterable[Tuple[float, float]] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        tau_grid = np.linspace(0.0, self.T, n_nodes)

        dual = AmericanPutBoundaryGK(
            K=1.0,
            r=self.q,
            q=self.r,
            sigma=self.sigma,
            T=self.T,
            gk_rule=self.gk_rule,
        )

        b0 = self.K / max(boundary_at_zero, 1e-12)
        _, B_dual, _ = dual.compute_boundary_with_terminal(
            n_nodes=n_nodes,
            n_iter=n_iter,
            boundary_at_zero=b0,
            tau_offset=tau_offset,
            div_schedule=div_schedule,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            S_star = self.K / B_dual

        S_star = np.maximum(S_star, self.K)
        S_star[0] = boundary_at_zero
        boundary_abs = (tau_grid + tau_offset, S_star)
        eep_local = self.continuation_call(
            boundary_at_zero,
            tau_offset,
            boundary=boundary_abs,
            div_schedule=div_schedule,
        ) - self.european_with_discrete_dividends("call", boundary_at_zero, tau_offset, div_schedule or [])

        return tau_grid, S_star, float(eep_local)

    def _merge_segments(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        tau_all: List[np.ndarray] = []
        B_all: List[np.ndarray] = []

        for tau_seg, B_seg in segments:
            tau_all.append(tau_seg)
            B_all.append(B_seg)

        tau = np.concatenate(tau_all)
        B = np.concatenate(B_all)

        order = np.argsort(tau, kind="mergesort")
        tau = tau[order]
        B = B[order]

        unique_tau, idx = np.unique(tau, return_index=True)
        B = B[idx]
        return unique_tau, B

    def _solve_dividend_boundary(
        self,
        option_type: str,
        t_div: float,
        D: float,
        boundary_after: Tuple[np.ndarray, np.ndarray],
        remaining_dividends: Iterable[Tuple[float, float]],
        eep_const: float,
        max_iter: int = 120,
        verbose: bool = False,
        eep_grid_size: int = 80,
    ) -> float:
        tau = self.T - t_div

        tau_after, boundary_after_grid = boundary_after
        boundary_after_func = self._interp_boundary(tau_after, boundary_after_grid)
        boundary_after_tau = boundary_after_func(tau)

        def european_raw(S: float) -> float:
            S_adj = max(S - D, 0.0)
            if option_type == "call":
                return self.european_with_discrete_dividends("call", S_adj, tau, remaining_dividends)
            return self.european_with_discrete_dividends("put", S_adj, tau, remaining_dividends)

        def make_interpolator(x: np.ndarray, y: np.ndarray) -> Callable[[float], float]:
            def interp_fn(z: float) -> float:
                return float(np.interp(z, x, y))

            return interp_fn

        def bs_delta(S_eff: float, tau_eff: float, is_call: bool) -> float:
            if tau_eff <= 0.0 or S_eff <= 0.0:
                return 1.0 if is_call else -1.0
            d1 = (log(S_eff / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * tau_eff) / (
                self.sigma * sqrt(tau_eff)
            )
            if is_call:
                return exp(-self.q * tau_eff) * norm_cdf(d1)
            return -exp(-self.q * tau_eff) * norm_cdf(-d1)

        if option_type == "call":
            low = self.K
            high = max(self.K * 2.0, self.K + max(D, 1e-6))

            if np.isfinite(boundary_after_tau):
                high = min(high, boundary_after_tau + D)

            if high <= low:
                high = low * 1.5
        else:
            low = 0.0
            high = self.K
            if np.isfinite(boundary_after_tau):
                low = max(low, boundary_after_tau + D)

            if high <= low:
                high = low * 1.5

        grid_size = max(16, int(eep_grid_size))
        if low > 0.0:
            s_grid = np.geomspace(low, high, grid_size)
        else:
            s_grid = np.linspace(low, high, grid_size)

        def f_sstar(S_star: float) -> float:
            euro = european_raw(S_star)
            if option_type == "call":
                return (S_star - self.K) - (euro + eep_const)
            return (self.K - S_star) - (euro + eep_const)

        def fprime_sstar(S_star: float) -> float:
            S_eff = max(S_star - D, 0.0)
            delta = bs_delta(S_eff, tau, option_type == "call")
            if option_type == "call":
                return 1.0 - delta
            return -1.0 - delta

        f_vals = np.array([f_sstar(s) for s in s_grid], dtype=float)
        bracket = None
        for j in range(1, len(s_grid)):
            f0, f1 = f_vals[j - 1], f_vals[j]
            if np.isfinite(f0) and np.isfinite(f1) and f0 * f1 <= 0.0:
                bracket = (s_grid[j - 1], s_grid[j])
                break

        if bracket is None:
            f_low = f_vals[0]
            f_high = f_vals[-1]
            if option_type == "call":
                if f_low >= 0.0:
                    return low
                return high
            else:
                if f_high <= 0.0:
                    return high
                return low

        low, high = bracket
        f_low = f_sstar(low)
        f_high = f_sstar(high)
        tol_f = 1e-10
        if abs(f_low) <= tol_f:
            return low
        if abs(f_high) <= tol_f:
            return high

        s = 0.5 * (low + high)
        for _ in range(max_iter):
            f_val = f_sstar(s)
            if abs(f_val) < tol_f:
                if verbose:
                    euro = european_raw(s)
                    print(
                        f"  S*={s:.8f}, smooth_paste_residual={f_val:.3f}, "
                        f"Euro={euro:.8f}, EEP={eep_const:.8f}, S-K={s - self.K:.8f}"
                    )
                return s

            fprime = fprime_sstar(s)
            if fprime != 0.0:
                s_new = s - f_val / fprime
            else:
                s_new = 0.5 * (low + high)

            if s_new <= low or s_new >= high or not np.isfinite(s_new):
                s_new = 0.5 * (low + high)

            if f_val > 0.0:
                high = s
            else:
                low = s
            s = s_new

            if abs(high - low) <= 1e-10 * max(1.0, abs(high) + abs(low)):
                break

        if verbose:
            euro = european_raw(s)
            residual = f_sstar(s)
            print(
                f"  S*={s:.8f}, smooth_paste_residual={residual:.3f}, "
                f"Euro={euro:.8f}, EEP={eep_const:.8f}, S-K={s - self.K:.8f}"
            )
        return s

    def compute_boundary_discrete_dividends(
        self,
        option_type: str,
        discrete_dividends: Iterable[Tuple[float, float]],
        n_nodes: int = 60,
        n_iter: int = 8,
        use_duality_for_call: bool = True,
        verbose: bool = False,
        eep_grid_size: int = 80,
        segment_info: List[dict] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        option = option_type.lower()
        if option not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")

        divs = sorted(
            [(float(t), float(D)) for t, D in discrete_dividends if 0.0 < t < self.T],
            key=lambda x: x[0],
        )

        times = [0.0] + [t for t, _ in divs] + [self.T]
        div_map = {t: D for t, D in divs}

        segments: List[Tuple[np.ndarray, np.ndarray]] = []
        boundary_after: Tuple[np.ndarray, np.ndarray] | None = None

        # Start from the last segment (nearest to expiry)
        last_start = times[-2]
        last_end = times[-1]
        seg_T = last_end - last_start
        if verbose:
            print(f"Segment [{last_start:.6f}, {last_end:.6f}] (near expiry), T_seg={seg_T:.6f}")
        div_schedule = [(t - last_start, d) for t, d in divs if t > last_start]
        if option == "put":
            seg_model = AmericanPutBoundaryGK(
                K=self.K,
                r=self.r,
                q=self.q,
                sigma=self.sigma,
                T=seg_T,
                gk_rule=self.gk_rule,
            )
            tau_local, B_local, eep_after = seg_model.compute_boundary_with_terminal(
                n_nodes=n_nodes,
                n_iter=n_iter,
                boundary_at_zero=self.K,
                tau_offset=self.T - last_end,
                div_schedule=div_schedule,
            )
        else:
            seg_model = AmericanPutBoundaryGK(
                K=self.K,
                r=self.r,
                q=self.q,
                sigma=self.sigma,
                T=seg_T,
                gk_rule=self.gk_rule,
            )
            if use_duality_for_call:
                tau_local, B_local, eep_after = seg_model.compute_call_boundary_dual_with_terminal(
                    n_nodes=n_nodes,
                    n_iter=n_iter,
                    boundary_at_zero=self.K,
                    tau_offset=self.T - last_end,
                    div_schedule=div_schedule,
                )
            else:
                tau_local, B_local, eep_after = seg_model.compute_call_boundary_direct_with_terminal(
                    n_nodes=n_nodes,
                    n_iter=n_iter,
                    boundary_at_zero=self.K,
                    tau_offset=self.T - last_end,
                    div_schedule=div_schedule,
                )

        offset = self.T - last_end
        tau_global = tau_local + offset
        segments.append((tau_global, B_local))
        boundary_after = (tau_global, B_local)
        if verbose:
            print(
                f"  segment boundary: B(t_start)={B_local[0]:.8f}, B(t_end)={B_local[-1]:.8f}"
            )
        if segment_info is not None:
            segment_info.append(
                {
                    "t_start": last_start,
                    "t_end": last_end,
                    "tau_start": offset,
                    "tau_end": offset + seg_T,
                    "B_start": float(B_local[0]),
                    "B_end": float(B_local[-1]),
                    "dividend": 0.0,
                    "S_star": None,
                }
            )

        # Walk backward through remaining segments
        for idx in range(len(times) - 2, 0, -1):
            t_end = times[idx]
            t_start = times[idx - 1]
            D = div_map.get(t_end, 0.0)

            if verbose:
                print(f"Segment [{t_start:.6f}, {t_end:.6f}], T_seg={t_end - t_start:.6f}, dividend D={D:.6f}")

            remaining_divs = [(t - t_end, d) for t, d in divs if t > t_end]
            S_star = self._solve_dividend_boundary(
                option,
                t_end,
                D,
                boundary_after,
                remaining_dividends=remaining_divs,
                eep_const=eep_after,
                verbose=verbose,
                eep_grid_size=eep_grid_size,
            )
            if verbose:
                print(f"  solved S* at t={t_end:.6f}: {S_star:.8f}")

            seg_T = t_end - t_start
            seg_model = AmericanPutBoundaryGK(
                K=self.K,
                r=self.r,
                q=self.q,
                sigma=self.sigma,
                T=seg_T,
                gk_rule=self.gk_rule,
            )

            div_schedule = [(t - t_start, d) for t, d in divs if t > t_start]
            if option == "put":
                tau_local, B_local, eep_after = seg_model.compute_boundary_with_terminal(
                    n_nodes=n_nodes,
                    n_iter=n_iter,
                    boundary_at_zero=S_star,
                    tau_offset=self.T - t_end,
                    div_schedule=div_schedule,
                )
            else:
                if use_duality_for_call:
                    tau_local, B_local, eep_after = seg_model.compute_call_boundary_dual_with_terminal(
                        n_nodes=n_nodes,
                        n_iter=n_iter,
                        boundary_at_zero=S_star,
                        tau_offset=self.T - t_end,
                        div_schedule=div_schedule,
                    )
                else:
                    tau_local, B_local, eep_after = seg_model.compute_call_boundary_direct_with_terminal(
                        n_nodes=n_nodes,
                        n_iter=n_iter,
                        boundary_at_zero=S_star,
                        tau_offset=self.T - t_end,
                        div_schedule=div_schedule,
                    )

            offset = self.T - t_end
            tau_global = tau_local + offset
            segments.insert(0, (tau_global, B_local))
            boundary_after = self._merge_segments(segments)
            if verbose:
                print(
                    f"  segment boundary: B(t_start)={B_local[0]:.8f}, B(t_end)={B_local[-1]:.8f}"
                )
            if segment_info is not None:
                segment_info.append(
                    {
                        "t_start": t_start,
                        "t_end": t_end,
                        "tau_start": offset,
                        "tau_end": offset + seg_T,
                        "B_start": float(B_local[0]),
                        "B_end": float(B_local[-1]),
                        "dividend": float(D),
                        "S_star": float(S_star),
                    }
                )

        return boundary_after

    def price_with_discrete_dividends(
        self,
        option_type: str,
        S: float,
        tau: float,
        discrete_dividends: Iterable[Tuple[float, float]],
        boundary: Tuple[np.ndarray, np.ndarray] | None = None,
        n_nodes: int = 60,
        n_iter: int = 8,
        use_duality_for_call: bool = True,
    ) -> float:
        if tau <= 0.0:
            return max(self.K - S, 0.0) if option_type.lower() == "put" else max(S - self.K, 0.0)

        option = option_type.lower()

        if boundary is None:
            boundary = self.compute_boundary_discrete_dividends(
                option_type=option,
                discrete_dividends=discrete_dividends,
                n_nodes=n_nodes,
                n_iter=n_iter,
                use_duality_for_call=use_duality_for_call,
            )

        breakdown = self.price_with_discrete_dividends_breakdown(
            option_type=option,
            S=S,
            tau=tau,
            discrete_dividends=discrete_dividends,
            boundary=boundary,
            n_nodes=n_nodes,
            n_iter=n_iter,
            use_duality_for_call=use_duality_for_call,
        )
        total, _, _ = cast(Tuple[float, float, List[float]], breakdown)
        return total

    def price_with_discrete_dividends_breakdown(
        self,
        option_type: str,
        S: float,
        tau: float,
        discrete_dividends: Iterable[Tuple[float, float]],
        boundary: Tuple[np.ndarray, np.ndarray] | None = None,
        segment_info: List[dict] | None = None,
        n_nodes: int = 60,
        n_iter: int = 8,
        use_duality_for_call: bool = True,
        return_components: bool = False,
        debug_eep_segment_index: int | None = None,
        debug_eep_samples: int = 6,
    ) -> Tuple[float, float, List[float]] | Tuple[float, float, List[float], List[dict[str, float]]]:
        if tau <= 0.0:
            price = max(self.K - S, 0.0) if option_type.lower() == "put" else max(S - self.K, 0.0)
            return price, price, []

        option = option_type.lower()

        if boundary is None:
            boundary = self.compute_boundary_discrete_dividends(
                option_type=option,
                discrete_dividends=discrete_dividends,
                n_nodes=n_nodes,
                n_iter=n_iter,
                use_duality_for_call=use_duality_for_call,
                segment_info=segment_info,
            )

        tau_grid, B_grid = boundary
        B_func = self._interp_boundary(tau_grid, B_grid)
        B_tau = B_func(tau)

        div_schedule = sorted(
            [(float(t), float(D)) for t, D in discrete_dividends if 0.0 < t <= tau],
            key=lambda x: x[0],
        )

        def effective_spot_and_q(s: float) -> Tuple[float, float, float]:
            if not div_schedule:
                return S, self.q, 0.0
            shift_s = self._future_div_shift_elapsed(s, div_schedule)
            shift_xi = self._future_div_shift_elapsed(tau - s, div_schedule)
            spot = max(S - shift_s, 0.0)
            return spot, self.q, shift_xi

        def d1_eff(S_eff: float, B: float, s: float, q_eff: float) -> float:
            if s <= 0.0:
                return float("inf") if S_eff > B else float("-inf")
            if S_eff <= 0.0 or B <= 0.0:
                return float("-inf")
            return (log(S_eff / B) + (self.r - q_eff + 0.5 * self.sigma**2) * s) / (
                self.sigma * sqrt(s)
            )

        def d2_eff(S_eff: float, B: float, s: float, q_eff: float) -> float:
            return d1_eff(S_eff, B, s, q_eff) - self.sigma * sqrt(s)

        def component_integrands(s: float) -> Tuple[float, float]:
            return 0.0, 0.0

        if option == "put":
            if S <= B_tau:
                price = self.K - S
                return price, price, []
            euro = self.european_with_discrete_dividends("put", S, tau, discrete_dividends)

            def integrand_s(s: float) -> float:
                B_xi = B_func(max(tau - s, 0.0))
                S_eff, q_eff, shift_xi = effective_spot_and_q(s)
                B_eff = max(B_xi - shift_xi, 0.0)
                d2 = d2_eff(S_eff, B_eff, s, q_eff)
                term = self.r * self.K * exp(-self.r * s) * norm_cdf(-d2)
                if q_eff > 0.0:
                    d1 = d1_eff(S_eff, B_eff, s, q_eff)
                    term -= q_eff * S_eff * exp(-q_eff * s) * norm_cdf(-d1)
                return term

            def component_integrands(s: float) -> Tuple[float, float]:
                B_xi = B_func(max(tau - s, 0.0))
                S_eff, q_eff, shift_xi = effective_spot_and_q(s)
                B_eff = max(B_xi - shift_xi, 0.0)
                d2 = d2_eff(S_eff, B_eff, s, q_eff)
                term_rk = self.r * self.K * exp(-self.r * s) * norm_cdf(-d2)
                term_qs = 0.0
                if q_eff > 0.0:
                    d1 = d1_eff(S_eff, B_eff, s, q_eff)
                    term_qs = -q_eff * S_eff * exp(-q_eff * s) * norm_cdf(-d1)
                return term_rk, term_qs
        else:
            if self.q <= 0.0 and boundary is None:
                euro = self.european_with_discrete_dividends("call", S, tau, discrete_dividends)
                return euro, euro, []
            if S >= B_tau:
                price = S - self.K
                return price, price, []
            euro = self.european_with_discrete_dividends("call", S, tau, discrete_dividends)

            def integrand_s(s: float) -> float:
                B_xi = B_func(max(tau - s, 0.0))
                S_eff, q_eff, shift_xi = effective_spot_and_q(s)
                B_eff = max(B_xi - shift_xi, 0.0)
                d2 = d2_eff(S_eff, B_eff, s, q_eff)
                term = -self.r * self.K * exp(-self.r * s) * norm_cdf(d2)
                if q_eff > 0.0:
                    d1 = d1_eff(S_eff, B_eff, s, q_eff)
                    term += q_eff * S_eff * exp(-q_eff * s) * norm_cdf(d1)
                return term

            def component_integrands(s: float) -> Tuple[float, float]:
                B_xi = B_func(max(tau - s, 0.0))
                S_eff, q_eff, shift_xi = effective_spot_and_q(s)
                B_eff = max(B_xi - shift_xi, 0.0)
                d2 = d2_eff(S_eff, B_eff, s, q_eff)
                term_rk = -self.r * self.K * exp(-self.r * s) * norm_cdf(d2)
                term_qs = 0.0
                if q_eff > 0.0:
                    d1 = d1_eff(S_eff, B_eff, s, q_eff)
                    term_qs = q_eff * S_eff * exp(-q_eff * s) * norm_cdf(d1)
                return term_rk, term_qs

        if segment_info is None:
            divs = sorted(
                [(float(t), float(D)) for t, D in discrete_dividends if 0.0 < t < self.T],
                key=lambda x: x[0],
            )
            times = [0.0] + [t for t, _ in divs] + [self.T]
            segment_info = []
            for i in range(1, len(times)):
                t_start, t_end = times[i - 1], times[i]
                segment_info.append(
                    {
                        "t_start": t_start,
                        "t_end": t_end,
                        "tau_start": self.T - t_end,
                        "tau_end": self.T - t_start,
                    }
                )

        if debug_eep_segment_index is not None and segment_info:
            ordered_segments = sorted(segment_info, key=lambda s: s["tau_start"])
            if 0 <= debug_eep_segment_index < len(ordered_segments):
                seg = ordered_segments[debug_eep_segment_index]
                s_start = max(0.0, float(seg["tau_start"]))
                s_end = min(tau, float(seg["tau_end"]))
                if s_end > s_start:
                    n_samples = max(2, int(debug_eep_samples))
                    grid = np.linspace(s_start, s_end, n_samples)
                    print("\nEEP integrand diagnostics:")
                    print(
                        f"  Segment idx={debug_eep_segment_index}, tau=[{s_start:.6f},{s_end:.6f}], samples={n_samples}"
                    )
                    for s_val in grid:
                        val = integrand_s(s_val)
                        comp_rk, comp_qs = component_integrands(s_val)
                        print(
                            f"  s={s_val:.6f} | integrand={val:.8e} | rK={comp_rk:.8e} | qS={comp_qs:.8e}"
                        )

        premiums: List[float] = []
        components: List[dict[str, float]] = []
        for seg in sorted(segment_info, key=lambda s: s["tau_start"]):
            s_start = max(0.0, float(seg["tau_start"]))
            s_end = min(tau, float(seg["tau_end"]))
            if s_end <= s_start:
                premiums.append(0.0)
                if return_components:
                    components.append({"rk": 0.0, "qs": 0.0})
                continue
            prem = self._gk_integrate(integrand_s, s_start, s_end)
            premiums.append(float(prem))
            if return_components:
                prem_rk = self._gk_integrate(lambda s: component_integrands(s)[0], s_start, s_end)
                prem_qs = self._gk_integrate(lambda s: component_integrands(s)[1], s_start, s_end)
                components.append({"rk": float(prem_rk), "qs": float(prem_qs)})

        total = euro + sum(premiums)
        if return_components:
            return total, euro, premiums, components
        return total, euro, premiums

    def european_with_discrete_dividends(
        self,
        option_type: str,
        S: float,
        tau: float,
        discrete_dividends: Iterable[Tuple[float, float]],
    ) -> float:
        if tau <= 0.0:
            return max(self.K - S, 0.0) if option_type.lower() == "put" else max(S - self.K, 0.0)

        divs = [
            (float(t), float(D))
            for t, D in discrete_dividends
            if 0.0 < t <= tau
        ]
        pv_div = sum(D * exp(-self.r * t) for t, D in divs)
        S_adj = max(S - pv_div, 0.0)

        if option_type.lower() == "put":
            return self.european_put(S_adj, tau)
        return self.european_call(S_adj, tau)

    def european_put(self, S: float, tau: float) -> float:
        if tau <= 0.0:
            return max(self.K - S, 0.0)
        d1 = self.d1(S, self.K, tau)
        d2 = d1 - self.sigma * sqrt(tau)
        return self.K * exp(-self.r * tau) * norm_cdf(-d2) - S * exp(-self.q * tau) * norm_cdf(-d1)

    def european_call(self, S: float, tau: float) -> float:
        if tau <= 0.0:
            return max(S - self.K, 0.0)
        d1 = self.d1(S, self.K, tau)
        d2 = d1 - self.sigma * sqrt(tau)
        return S * exp(-self.q * tau) * norm_cdf(d1) - self.K * exp(-self.r * tau) * norm_cdf(d2)

    def price_put(
        self,
        S: float,
        tau: float,
        boundary: Tuple[np.ndarray, np.ndarray] | None = None,
        n_nodes: int = 120,
        n_iter: int = 8,
    ) -> float:
        if tau <= 0.0:
            return max(self.K - S, 0.0)

        if boundary is None:
            tau_grid, B_grid = self.compute_boundary(n_nodes=n_nodes, n_iter=n_iter)
        else:
            tau_grid, B_grid = boundary

        B_func = self._interp_boundary(tau_grid, B_grid)
        B_tau = B_func(tau)
        if S <= B_tau:
            return self.K - S

        euro = self.european_put(S, tau)

        def integrand_s(s: float) -> float:
            B_xi = B_func(max(tau - s, 0.0))
            d2 = self.d2(S, B_xi, s)
            term = self.r * self.K * exp(-self.r * s) * norm_cdf(-d2)
            if self.q > 0.0:
                d1 = self.d1(S, B_xi, s)
                term -= self.q * S * exp(-self.q * s) * norm_cdf(-d1)
            return term

        prem = self._gk_integrate(integrand_s, 0.0, tau)
        return euro + prem

    def price_call(
        self,
        S: float,
        tau: float,
        boundary: Tuple[np.ndarray, np.ndarray] | None = None,
        n_nodes: int = 120,
        n_iter: int = 8,
    ) -> float:
        if tau <= 0.0:
            return max(S - self.K, 0.0)
        if self.q <= 0.0 and boundary is None:
            # No early exercise for calls without continuous dividends
            return self.european_call(S, tau)

        if boundary is None:
            tau_grid, S_star_grid = self.compute_call_boundary(n_nodes=n_nodes, n_iter=n_iter)
        else:
            tau_grid, S_star_grid = boundary

        S_star_func = self._interp_boundary(tau_grid, S_star_grid)
        S_star_tau = S_star_func(tau)
        if S >= S_star_tau:
            return S - self.K

        euro = self.european_call(S, tau)

        def integrand_s(s: float) -> float:
            B_xi = S_star_func(max(tau - s, 0.0))
            d2 = self.d2(S, B_xi, s)
            term = -self.r * self.K * exp(-self.r * s) * norm_cdf(d2)
            if self.q > 0.0:
                d1 = self.d1(S, B_xi, s)
                term += self.q * S * exp(-self.q * s) * norm_cdf(d1)
            return term

        prem = self._gk_integrate(integrand_s, 0.0, tau)
        return euro + prem

    def price_call_direct(
        self,
        S: float,
        tau: float,
        boundary: Tuple[np.ndarray, np.ndarray] | None = None,
        n_nodes: int = 120,
        n_iter: int = 8,
    ) -> float:
        if tau <= 0.0:
            return max(S - self.K, 0.0)
        if self.q <= 0.0:
            return self.european_call(S, tau)

        if boundary is None:
            tau_grid, S_star_grid = self.compute_call_boundary_direct(n_nodes=n_nodes, n_iter=n_iter)
        else:
            tau_grid, S_star_grid = boundary

        S_star_func = self._interp_boundary(tau_grid, S_star_grid)
        S_star_tau = S_star_func(tau)
        if S >= S_star_tau:
            return S - self.K

        euro = self.european_call(S, tau)

        def integrand_s(s: float) -> float:
            S_xi = S_star_func(max(tau - s, 0.0))
            d2 = self.d2(S, S_xi, s)
            term = -self.r * self.K * exp(-self.r * s) * norm_cdf(d2)
            d1 = self.d1(S, S_xi, s)
            term += self.q * S * exp(-self.q * s) * norm_cdf(d1)
            return term

        prem = self._gk_integrate(integrand_s, 0.0, tau)
        return euro + prem

    def continuation_put(
        self,
        S: float,
        tau: float,
        boundary: Tuple[np.ndarray, np.ndarray],
        div_schedule: Iterable[Tuple[float, float]] | None = None,
    ) -> float:
        if tau <= 0.0:
            return max(self.K - S, 0.0)

        tau_grid, B_grid = boundary
        B_func = self._interp_boundary(tau_grid, B_grid)

        if div_schedule is not None:
            euro = self.european_with_discrete_dividends("put", S, tau, div_schedule)
        else:
            euro = self.european_put(S, tau)

        def integrand_s(s: float) -> float:
            B_xi = B_func(max(tau - s, 0.0))
            if div_schedule is not None:
                shift_s = self._future_div_shift_elapsed(s, div_schedule)
                shift_xi = self._future_div_shift_elapsed(tau - s, div_schedule)
                S_eff = max(S - shift_s, 0.0)
                B_eff = max(B_xi - shift_xi, 0.0)
            else:
                S_eff = S
                B_eff = B_xi
            d2 = self.d2(S_eff, B_eff, s)
            term = self.r * self.K * exp(-self.r * s) * norm_cdf(-d2)
            if self.q > 0.0:
                d1 = self.d1(S_eff, B_eff, s)
                term -= self.q * S_eff * exp(-self.q * s) * norm_cdf(-d1)
            return term

        prem = self._gk_integrate(integrand_s, 0.0, tau)
        return euro + prem

    def continuation_call(
        self,
        S: float,
        tau: float,
        boundary: Tuple[np.ndarray, np.ndarray],
        div_schedule: Iterable[Tuple[float, float]] | None = None,
    ) -> float:
        if tau <= 0.0:
            return max(S - self.K, 0.0)
        if self.q <= 0.0:
            return self.european_call(S, tau)

        tau_grid, S_star_grid = boundary
        S_star_func = self._interp_boundary(tau_grid, S_star_grid)

        if div_schedule is not None:
            euro = self.european_with_discrete_dividends("call", S, tau, div_schedule)
        else:
            euro = self.european_call(S, tau)

        def integrand_s(s: float) -> float:
            B_xi = S_star_func(max(tau - s, 0.0))
            if div_schedule is not None:
                shift_s = self._future_div_shift_elapsed(s, div_schedule)
                shift_xi = self._future_div_shift_elapsed(tau - s, div_schedule)
                S_eff = max(S - shift_s, 0.0)
                B_eff = max(B_xi - shift_xi, 0.0)
            else:
                S_eff = S
                B_eff = B_xi
            d2 = self.d2(S_eff, B_eff, s)
            term = -self.r * self.K * exp(-self.r * s) * norm_cdf(d2)
            if self.q > 0.0:
                d1 = self.d1(S_eff, B_eff, s)
                term += self.q * S_eff * exp(-self.q * s) * norm_cdf(d1)
            return term

        prem = self._gk_integrate(integrand_s, 0.0, tau)
        return euro + prem


def plot_discrete_div_boundary_comparison(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: str,
    discrete_dividends: Iterable[Tuple[float, float]],
    n_nodes: int = 80,
    n_iter: int = 8,
    use_duality_for_call: bool = True,
    fd_M: int = 400,
    fd_N: int = 2000,
    buffer_eps: float = 1e-6,
    output_prefix: str = "boundary_output",
) -> None:
    import matplotlib.pyplot as plt
    import csv

    from fd_pricer import FiniteDifferenceAmerican

    model = AmericanPutBoundaryGK(K=K, r=r, q=q, sigma=sigma, T=T)
    segment_info: List[dict] = []
    tau_grid, boundary = model.compute_boundary_discrete_dividends(
        option_type=option_type,
        discrete_dividends=discrete_dividends,
        n_nodes=n_nodes,
        n_iter=n_iter,
        use_duality_for_call=use_duality_for_call,
        verbose=True,
        segment_info=segment_info,
    )

    debug_segment_index = 1 if len(segment_info) > 1 else None
    breakdown = model.price_with_discrete_dividends_breakdown(
        option_type=option_type,
        S=S0,
        tau=T,
        discrete_dividends=discrete_dividends,
        boundary=(tau_grid, boundary),
        segment_info=segment_info,
        n_nodes=n_nodes,
        n_iter=n_iter,
        use_duality_for_call=use_duality_for_call,
        return_components=True,
        debug_eep_segment_index=debug_segment_index,
        debug_eep_samples=6,
    )
    price_gk, euro_gk, seg_prem, seg_components = cast(
        Tuple[float, float, List[float], List[dict[str, float]]],
        breakdown,
    )

    fd = FiniteDifferenceAmerican(
        S0=S0,
        K=K,
        r=r,
        T=T,
        sigma=sigma,
        q=q,
        option_type=option_type,
        is_american=True,
        M=fd_M,
        N=fd_N,
        discrete_dividends=list(discrete_dividends),
    )
    price_fd, boundary_fd = fd.price(return_boundary=True)

    fd_euro = FiniteDifferenceAmerican(
        S0=S0,
        K=K,
        r=r,
        T=T,
        sigma=sigma,
        q=q,
        option_type=option_type,
        is_american=False,
        M=fd_M,
        N=fd_N,
        discrete_dividends=list(discrete_dividends),
    )
    price_fd_euro = fd_euro.price(return_boundary=False)

    t_gk = T - tau_grid
    # t_fd = [T - tau for tau, b in boundary_fd if b is not None]
    t_fd = [tau for tau, b in boundary_fd if b is not None]
    b_fd = [b for _, b in boundary_fd if b is not None]
    if t_fd:
        t_fd_arr = np.array(t_fd, dtype=float)
        b_fd_arr = np.array(b_fd, dtype=float)
        order_fd = np.argsort(t_fd_arr)
        t_fd_arr = t_fd_arr[order_fd]
        b_fd_arr = b_fd_arr[order_fd]
        b_fd_interp = np.interp(t_gk, t_fd_arr, b_fd_arr)
    else:
        t_fd_arr = np.array([], dtype=float)
        b_fd_arr = np.array([], dtype=float)
        b_fd_interp = np.array([], dtype=float)

    div_times = sorted([float(t) for t, _ in discrete_dividends if 0.0 < t < T])

    def buffer_times(t_vals: np.ndarray, segs: List[dict], eps: float) -> np.ndarray:
        if eps <= 0.0 or not segs:
            return t_vals
        t_buf = np.array(t_vals, dtype=float)
        for seg in segs:
            t_start = float(seg.get("t_start", 0.0))
            t_end = float(seg.get("t_end", 0.0))
            mask = (t_buf >= t_start - 1e-12) & (t_buf <= t_end + 1e-12)
            if not np.any(mask):
                continue
            if t_end in div_times:
                t_buf[mask] = np.minimum(t_buf[mask], t_end - eps)
            if t_start in div_times:
                t_buf[mask] = np.maximum(t_buf[mask], t_start + eps)
        return t_buf

    t_gk_plot = buffer_times(t_gk, segment_info, buffer_eps)

    plt.figure(figsize=(9, 5))
    plt.plot(t_gk_plot, boundary, label="GK boundary (piecewise)")
    if t_fd:
        plt.plot(t_gk_plot, b_fd_interp, label="FDM boundary (interp)", alpha=0.8)

    for t_div, _ in sorted([(float(t), float(D)) for t, D in discrete_dividends]):
        if 0.0 < t_div < T:
            plt.axvline(t_div, color="gray", linestyle="--", linewidth=0.8)

    critical_points = [
        (seg["t_end"], seg["S_star"]) for seg in segment_info if seg.get("S_star") is not None
    ]
    if critical_points:
        t_cp = [t for t, _ in critical_points]
        b_cp = [b for _, b in critical_points]
        plt.scatter(t_cp, b_cp, color="black", s=28, zorder=5, label="GK critical points")

    plt.xlabel("Time t")
    plt.ylabel("Critical price")
    plt.title(f"American {option_type} boundary with discrete dividends")
    plt.legend()
    plt.tight_layout()
    png_path = f"{output_prefix}.png"
    plt.savefig(png_path, dpi=160)
    plt.show()

    print("\nGK price breakdown:")
    print(f"  European (dividend-adjusted): {euro_gk:.8f}")
    ordered_segments = sorted(segment_info, key=lambda s: s.get("tau_start", 0.0))
    for seg, prem, comp in zip(ordered_segments, seg_prem, seg_components):
        seg_start = seg.get("t_start", 0.0)
        seg_end = seg.get("t_end", 0.0)
        tau_start = seg.get("tau_start", 0.0)
        tau_end = seg.get("tau_end", 0.0)
        b_start = seg.get("B_start")
        b_end = seg.get("B_end")
        s_star = seg.get("S_star")
        D = seg.get("dividend", 0.0)
        b_start_txt = "NA" if b_start is None else f"{b_start:.6f}"
        b_end_txt = "NA" if b_end is None else f"{b_end:.6f}"
        s_star_txt = "NA" if s_star is None else f"{s_star:.6f}"
        print(
            "  Segment "
            f"[{seg_start:.6f}, {seg_end:.6f}] "
            f"tau=[{tau_start:.6f},{tau_end:.6f}] "
            f"B_start={b_start_txt}, B_end={b_end_txt}, "
            f"D={D:.6f}, S*={s_star_txt}, "
            f"EEP={prem:.8f}, EEP_rK={comp['rk']:.8f}, EEP_qS={comp['qs']:.8f}"
        )
    print(f"  Total GK price: {price_gk:.8f}")
    print(f"FDM price: {price_fd:.8f}")
    print(f"FDM European: {price_fd_euro:.8f}")
    print(f"FDM EEP: {price_fd - price_fd_euro:.8f}")

    gk_csv = f"{output_prefix}_gk.csv"
    with open(gk_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "tau", "boundary"])
        for t_val, tau_val, b_val in zip(t_gk, tau_grid, boundary):
            writer.writerow([f"{t_val:.16f}", f"{tau_val:.16f}", f"{b_val:.16f}"])

    fdm_csv = f"{output_prefix}_fdm.csv"
    with open(fdm_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "tau", "boundary"])
        for (tau_val, b_val) in boundary_fd:
            if b_val is None:
                continue
            t_val = T - tau_val
            writer.writerow([f"{t_val:.16f}", f"{tau_val:.16f}", f"{b_val:.16f}"])

    fdm_interp_csv = f"{output_prefix}_fdm_interp.csv"
    if len(b_fd_interp) > 0:
        with open(fdm_interp_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "tau", "boundary_interp"])
            for t_val, tau_val, b_val in zip(t_gk, tau_grid, b_fd_interp):
                writer.writerow([f"{t_val:.16f}", f"{tau_val:.16f}", f"{b_val:.16f}"])

from ju_1998 import Ju1998Pricing

if __name__ == "__main__":
    model = AmericanPutBoundaryGK(K=100, r=0.01, q=0.02, sigma=0.1, T=1.0)
    n_nodes = 100
    tau_grid, boundary = model.compute_call_boundary(n_nodes=n_nodes, n_iter=6)
    #tau_grid_dir, boundary_dir = model.compute_call_boundary_direct(n_nodes=n_nodes, n_iter=10)

    
    ju = Ju1998Pricing(
        100, model.K, model.r, model.T, model.sigma, model.q, "call"
    )
    S_star = ju.solve_critical_price()
    print(f"Ju (1998) S*: {S_star:.6f}\n")


    for t, b in zip(tau_grid, boundary):
        #print(f"B({t:.3f}):\t{b:.6f}\t(direct: {b_dir:.6f})")
        print(f"{t:.4f}\t{b:.6f}")


    #for t, b in zip(tau_grid[::4], boundary[::4]):
    # for t, b, b_dir in zip(tau_grid, boundary, boundary_dir):
        #print(f"B({t:.3f}):\t{b:.6f}\t(direct: {b_dir:.6f})")
        # print(f"{t:.4f}\t{b:.6f}\t{b_dir:.6f}")