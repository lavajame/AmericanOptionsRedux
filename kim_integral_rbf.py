"""
Kim's Integral Approach for American Option Pricing in y-space.

Uses Gaussian RBF integration in y-space (y = sqrt(tau)) with Newton's method
to solve for the early exercise boundary simultaneously at all knots.

Key formulation:
  - Transform: y = sqrt(T - t), y_max = sqrt(T)
  - Log boundary: log(B(y)/K) at N knots, initialized via design matrix
  - Design matrix: asymptotic powers y^k/(c^k + y^k) + step functions at dividends
  - Integration: Gaussian RBF operator with analytic antiderivatives
  - Solver: Newton's method with lower-triangular analytic Jacobian

Supports:
  - American puts (w=-1) and calls (w=+1)
  - Continuous dividend yield q
  - Discrete dividends at specified times
"""

import time
import numpy as np
# import pandas as pd
from scipy.stats import norm
from scipy.special import erf


# ---------------------------------------------------------------------------
# Gaussian RBF helpers
# ---------------------------------------------------------------------------

def _gaussian_rbf(r, shape):
    """phi(r) = exp(-shape * r^2)"""
    return np.exp(-shape * r ** 2)


def _gaussian_rbf_antideriv(z, shape):
    """G(z) = int_0^z exp(-shape * t^2) dt = sqrt(pi)/(2*sqrt(shape)) * erf(sqrt(shape)*z)"""
    sq = np.sqrt(shape)
    return np.sqrt(np.pi) / (2 * sq) * erf(sq * z)


# ---------------------------------------------------------------------------
# Ju (1998) warm-start
# ---------------------------------------------------------------------------

def _ju_critical_price(K, T, r, q, sigma, w) -> float:
    """Solve for Ju (1998) critical stock price S* at maturity for warm start."""
    phi = w
    h = 1 - np.exp(-r * T)
    alpha = 2 * r / sigma ** 2
    beta = 2 * (r - q) / sigma ** 2
    term = (beta - 1) ** 2 + 4 * alpha / max(h, 1e-7)
    lam = (-(beta - 1) + phi * np.sqrt(term)) / 2

    def objective(S_star):
        d1 = (np.log(S_star / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        ve = phi * (
            S_star * np.exp(-q * T) * norm.cdf(phi * d1)
            - K * np.exp(-r * T) * norm.cdf(phi * d2)
        )
        prob = phi * np.exp(-q * T) * norm.cdf(phi * d1)
        rhs = prob + lam * (phi * (S_star - K) - ve) / S_star
        return phi - rhs

    # Set initial bounds for bisection method
    if w == +1:
        lo, hi = K*0.5, K * 2.0
    else:
        lo, hi = 1e-4, K*1.5

    # Ensure the objective function changes sign at the bounds
    f_lo = objective(lo)
    f_hi = objective(hi)

    if np.sign(f_lo) == np.sign(f_hi):
        print(f"Warning: Ju critical price objective has same sign at bounds: f({lo})={f_lo:.4e}, f({hi})={f_hi:.4e}")
        raise ValueError("Bisection method requires that the objective function has opposite signs at the endpoints.")

    tol = 1e-7  # Tolerance for convergence
    max_iter = 1000  # Maximum number of iterations

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = objective(mid)

        if abs(f_mid) < tol:
            return mid

        if np.sign(f_lo) == np.sign(f_mid):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid

    raise RuntimeError("Bisection method did not converge within the maximum number of iterations.")


# ===========================================================================
# Main solver class
# ===========================================================================

class KimIntegralRBF:
    """
    American option pricer using Kim's integral equation in y-space with
    Gaussian RBF integration and Newton iteration.

    Parameters
    ----------
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free interest rate.
    q : float
        Continuous dividend yield.
    sigma : float
        Volatility.
    w : int
        +1 for call, -1 for put.
    N : int
        Number of evenly spaced y-knots (before adding dividend knots).
    n_powers : int
        Number of asymptotic power terms in the design matrix.
    div_times : list or None
        Calendar times of discrete dividends (0 < t < T).
    div_amounts : list or None
        Corresponding dividend amounts.
    """

    def __init__(self, K, T, r, q, sigma, w=-1, N=16, n_powers=2,
                 div_times=None, div_amounts=None):
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.q = float(q)
        self.sigma = float(sigma)
        self.w = int(w)
        self.n_powers = int(n_powers)
        self.y_max = np.sqrt(T)

        # ---- 1. Discrete dividends ----
        if div_times is not None and div_amounts is not None:
            self.div_schedule = sorted(
                [(float(t), float(d)) for t, d in zip(div_times, div_amounts)
                 if 0 < t < T],
                key=lambda x: x[0],
            )
        else:
            self.div_schedule = []
        self.div_y = [np.sqrt(T - t) for t, _ in self.div_schedule]

        # ---- 2. Generate y-knots ----
        self.y = self._generate_y_knots(N)
        self.N = len(self.y)

        # ---- 3. Design matrix for log-boundary representation ----
        self.A_design = self._build_design_matrix()

        # ---- 4. Gaussian RBF integration operator ----
        self._build_rbf_operator()

        # Boundary at y=0: B(0) = K
        self.B0 = self.K

    # ------------------------------------------------------------------
    # 2. y-knot generation
    # ------------------------------------------------------------------
    def _generate_y_knots(self, N):
        """
        Generate evenly spaced y-knots across [0, y_max].
        Include knots at y_div +/- eps for each discrete dividend time to
        capture boundary discontinuities.
        """
        y_base = np.linspace(0, self.y_max, N)
        extra = []
        eps = 1e-5
        for y_d in self.div_y:
            if y_d - eps > 0:
                extra.append(y_d - eps)
            if y_d + eps < self.y_max:
                extra.append(y_d + eps)
        if extra:
            y_all = np.sort(np.unique(np.concatenate([y_base, extra])))
        else:
            y_all = y_base
        return y_all[(y_all >= 0) & (y_all <= self.y_max)]

    # ------------------------------------------------------------------
    # 3. Design matrix
    # ------------------------------------------------------------------
    def _build_design_matrix(self):
        """
        Build design matrix A so that log(B(y)/K) = A @ c.

        Columns:
          k=1..n_powers : y^k / (c^k + y^k)   (asymptotic powers, c = y_max)
          i=1..n_div    : H(y - y_div_i)       (step functions at dividend times)

        All basis functions equal 0 at y=0 so B(0) = K is automatic.
        Powers approach 1 as y -> inf, capturing the long-maturity asymptote.
        """
        y = self.y
        c = self.y_max
        n = len(y)

        power_cols = np.zeros((n, self.n_powers))
        for k in range(1, self.n_powers + 1):
            power_cols[:, k - 1] = y ** k / (c ** k + y ** k)

        step_cols = np.zeros((n, len(self.div_schedule)))
        for i, y_d in enumerate(self.div_y):
            # Step function is active for y >= y_d
            # Captures discontinuity at dividend time
            step_cols[:, i] = (y >= y_d).astype(float)

        return np.hstack([power_cols, step_cols]) if step_cols.shape[1] > 0 else power_cols

    # ------------------------------------------------------------------
    # 4. Gaussian RBF integration operator
    # ------------------------------------------------------------------
    def _build_rbf_operator(self):
        """
        Build RBF operator matrix for numerical integration.

        Gaussian kernel:  phi(r) = exp(-shape * r^2)
        Antiderivative:   G(z) = sqrt(pi)/(2*sqrt(shape)) * erf(sqrt(shape)*z)

        Operator = Psi @ inv(Phi^T Phi + alpha I) @ Phi^T

        Phi_ij = phi(|y_i - y_j|)           -- interpolation matrix
        Psi_ij = G(y_i - y_j) - G(-y_j)     -- integration matrix (0 to y_i)
        """
        y = self.y
        n = len(y)
        dy = np.mean(np.diff(y)) if n > 1 else 1.0
        self.rbf_shape = 1.0 / (2.0 * dy ** 2)
        alpha_reg = 1e-11

        R = np.abs(y[:, None] - y[None, :])
        Phi = _gaussian_rbf(R, self.rbf_shape)

        upper = y[:, None] - y[None, :]
        lower = -y[None, :]
        Psi = _gaussian_rbf_antideriv(upper, self.rbf_shape) \
            - _gaussian_rbf_antideriv(lower, self.rbf_shape)

        self.A_rbf = Psi @ np.linalg.solve(
            Phi.T @ Phi + alpha_reg * np.eye(n), Phi.T
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _pv_divs(self, tau):
        """PV of discrete dividends occurring at or after calendar time T - tau."""
        t_now = self.T - tau
        pv = 0.0
        for t_d, D in self.div_schedule:
            if t_d >= t_now:
                pv += D * np.exp(-self.r * (t_d - t_now))
        return pv

    def _bs(self, S, K, tau):
        """Black-Scholes European price and delta for option type w."""
        tau = max(tau, 1e-12)
        vt = self.sigma * np.sqrt(tau)
        d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma ** 2) * tau) / vt
        d2 = d1 - vt
        eq, er = np.exp(-self.q * tau), np.exp(-self.r * tau)
        price = self.w * (S * eq * norm.cdf(self.w * d1)
                          - K * er * norm.cdf(self.w * d2))
        delta = self.w * eq * norm.cdf(self.w * d1)
        return price, delta

    def _bisection(self, objective, lo, hi, tol=1e-7, max_iter=1000):
        """Generic bisection solver for f(x) = 0."""
        f_lo = objective(lo)
        f_hi = objective(hi)

        if np.sign(f_lo) == np.sign(f_hi):
            raise ValueError(
                f"Bisection requires opposite signs at bounds: "
                f"f({lo})={f_lo:.4e}, f({hi})={f_hi:.4e}"
            )

        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            f_mid = objective(mid)

            if abs(f_mid) < tol:
                return mid

            if np.sign(f_lo) == np.sign(f_mid):
                lo, f_lo = mid, f_mid
            else:
                hi, f_hi = mid, f_mid

        raise RuntimeError(
            "Bisection did not converge within maximum iterations."
        )

    def _initial_log_boundary(self):
        """
        Initial log(B(y)/K) at each y-knot.

        Uses Ju (1998) critical price S* at T to linearly interpolate
        log(B/K) from 0 (at y=0) to log(S*/K) (at y=y_max).

        If discrete dividends are present, adjusts the boundary at each
        dividend time t_d by solving w*(S*-K) = Euro(S*-sumDivs, K, tau_d)
        and scaling the boundary after t_d by S* / B(y_d - eps).
        Dividends are processed from closest to expiry (largest t_d) backwards,
        accumulating their effects as we move back in time.
        """
        # Start with Ju's linear interpolation
        S_star = _ju_critical_price(self.K, self.T, self.r, self.q,
                                    self.sigma, self.w)
        
        print(f"Ju critical price S* at maturity: {S_star:.4f}")

        log_BK_max = np.log(S_star / self.K)
        log_BK = log_BK_max * self.y / self.y_max

        # If we have discrete dividends, adjust boundary at each dividend
        if self.div_schedule:
            B = self.K * np.exp(log_BK)

            # Process dividends from closest to expiry to furthest
            # (i.e., from largest t_d to smallest t_d, or smallest y_d to largest y_d)
            # This ensures we account for cumulative effects as we go back in time
            for t_d, D in sorted(self.div_schedule, key=lambda x: x[0], reverse=True):
                y_d = np.sqrt(self.T - t_d)
                tau_d = self.T - t_d

                # Find S* at t_d: w*(S*-K) = Euro(S*-sumDivs, K, tau_d)
                # sumDivs includes all dividends AFTER t_d (which we've already processed)
                pv_divs = self._pv_divs(tau_d)

                def objective(S_star):
                    S_adj = max(S_star - pv_divs, 1e-8)
                    euro_price, _ = self._bs(S_adj, self.K, tau_d)
                    return self.w * (S_star - self.K) - euro_price

                # Solve for S* using bisection
                if self.w == +1:  # call
                    lo, hi = self.K * 0.5, self.K * 3.0
                else:  # put
                    lo, hi = 1e-4, self.K * 1.5

                try:
                    S_star_d = self._bisection(objective, lo, hi)
                except (ValueError, RuntimeError):
                    # If bisection fails, skip this adjustment
                    continue
                
                print(f"Dividend at t={t_d:.3f}: S*={S_star_d:.4f}, PV(divs)={pv_divs:.4f}")

                # Find B just before the dividend (y slightly less than y_d)
                # B_before already includes effects of all later dividends (processed earlier)
                idx_before = np.searchsorted(self.y, y_d) - 1
                if idx_before >= 0 and idx_before < len(B):
                    B_before = B[idx_before]
                    if B_before > 1e-8:  # avoid division by zero
                        # Compute jump ratio
                        ratio = S_star_d / B_before

                        # Scale all boundary values at y >= y_d by this ratio
                        # (i.e., from this dividend forward to maturity)
                        mask = self.y >= y_d
                        B[mask] *= ratio

            # Convert back to log space
            log_BK = np.log(B / self.K)

        return log_BK

    # ------------------------------------------------------------------
    # 5 & 6. Residual (smooth pasting) and Jacobian
    # ------------------------------------------------------------------
    def get_residual_and_jac(self, H):
        """
        Compute residual and lower-triangular Jacobian.

        Parameters
        ----------
        H : array, length N-1
            log(B(y_k)/K) for k=1,...,N-1.  Knot 0 is fixed (B(0)=K).

        Returns
        -------
        res : array, length N-1
            Smooth-pasting residual: w*(B_k - K) - (Euro_k + EEP_k)
        jac : (N-1) x (N-1) lower-triangular matrix
            d(res) / d(log(B/K))
        """
        B = np.concatenate([[self.B0], self.K * np.exp(H)])
        N = len(B)
        y = self.y

        # --- tau matrix and validity mask ---
        tau_mat = y[:, None] ** 2 - y[None, :] ** 2
        valid = tau_mat > 1e-12
        tau_safe = np.where(valid, tau_mat, 1e-12)

        # --- Black-Scholes quantities on the (k, j) grid ---
        vt = self.sigma * np.sqrt(tau_safe)
        log_ratio = np.where(valid,
                             np.log(np.maximum(B[:, None] / B[None, :], 1e-30)),
                             0.0)
        d1 = np.where(valid,
                      (log_ratio + (self.r - self.q + 0.5 * self.sigma ** 2)
                       * tau_safe) / vt,
                      0.0)
        d2 = d1 - vt

        eq = np.exp(-self.q * tau_safe)
        er = np.exp(-self.r * tau_safe)

        Nd1 = norm.cdf(self.w * d1)
        Nd2 = norm.cdf(self.w * d2)
        nd1 = norm.pdf(d1)
        nd2 = norm.pdf(d2)

        # --- EEP integrand matrix (lower triangular: k > j) ---
        # f[k,j] = 2*y_j * w * (q*B_k*e^{-q*tau}*N(w*d1) - r*K*e^{-r*tau}*N(w*d2))
        f_mat = np.where(valid,
                         2 * y[None, :] * self.w * (
                             self.q * B[:, None] * eq * Nd1
                             - self.r * self.K * er * Nd2
                         ), 0.0)
        # zero upper triangle + diagonal
        f_mat[np.triu_indices(N)] = 0.0

        # --- EEP via RBF operator ---
        eep = np.sum(self.A_rbf * f_mat, axis=1)

        # --- European price and delta at each knot ---
        euro_p = np.zeros(N)
        euro_d = np.zeros(N)
        for k in range(N):
            tau_k = y[k] ** 2
            pv = self._pv_divs(tau_k)
            S_adj = max(B[k] - pv, 1e-8)
            p, d = self._bs(S_adj, self.K, tau_k)
            euro_p[k] = p
            euro_d[k] = d

        # ---- 5. Smooth pasting residual ----
        res = (self.w * (B - self.K) - (euro_p + eep))[1:]

        # ---- 6. Jacobian (lower triangular) ----
        dd1_dS = np.where(valid, 1.0 / (B[:, None] * vt), 0.0)
        dd1_dB = np.where(valid, -1.0 / (B[None, :] * vt), 0.0)

        # common pdf factor
        pdf_term = self.q * B[:, None] * eq * nd1 - self.r * self.K * er * nd2

        # df/dB_k (spot derivative) -- affects diagonal
        df_dS = np.where(valid,
                         2 * y[None, :] * self.w * (
                             self.q * eq * Nd1
                             + self.w * dd1_dS * pdf_term
                         ), 0.0)
        df_dS[np.triu_indices(N)] = 0.0

        # df/dB_j (strike derivative) -- affects off-diagonal
        df_dB = np.where(valid,
                         2 * y[None, :] * self.w * self.w * dd1_dB * pdf_term,
                         0.0)
        df_dB[np.triu_indices(N)] = 0.0

        # Assemble J_lin[k, j] = d(res_k)/d(B_j)
        # Off-diagonal: -A_rbf[k,j] * df_dB[k,j]
        J_lin = -self.A_rbf * df_dB
        J_lin[np.triu_indices(N)] = 0.0

        # Diagonal: w - euro_delta - sum_j A_rbf[k,j]*df_dS[k,j]
        diag_eep = np.sum(self.A_rbf * df_dS, axis=1)
        np.fill_diagonal(J_lin, self.w - euro_d - diag_eep)

        # Convert from d/dB to d/d(log(B/K)) via chain rule
        J_log = J_lin[1:, 1:] * B[1:][None, :]

        return res, J_log

    # ------------------------------------------------------------------
    # 7. Newton solver
    # ------------------------------------------------------------------
    def solve_boundary(self, max_iters=5, tol=1e-9):
        """
        Solve for the early exercise boundary via Newton's method.

        Returns
        -------
        B : array, length N
            Boundary values at y-knots.
        history : dict
            Iteration snapshots {label: (B_array, rmse)}.
        initial_residual_norm : float
            Initial residual norm (before Newton iterations).
        """
        log_BK = self._initial_log_boundary()
        H = log_BK[1:]  # unknowns (knot 0 is fixed)

        # Compute initial residual norm for diagnostics
        res_init, _ = self.get_residual_and_jac(H)
        initial_residual_norm = np.linalg.norm(res_init)

        history = {}
        for it in range(max_iters):
            res, jac = self.get_residual_and_jac(H)
            res_rmse = np.sqrt(np.sum(res ** 2))
            
            history[f"Iter{it:2d}"] = (
                np.concatenate([[self.B0], self.K * np.exp(H)]),
                res_rmse
            )

            res_norm = np.linalg.norm(res)
            if res_norm < tol:
                break

            # Newton step: solve J * delta = res
            try:
                delta = np.linalg.solve(jac, res)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(jac, res, rcond=None)[0]

            H -= delta

            # Clip to reasonable range
            if self.w == -1:                             # put: B <= K
                H = np.clip(H, np.log(0.01), np.log(2.0))
            else:                                        # call: B >= K
                H = np.clip(H, np.log(0.5), np.log(10.0))

        B_final = np.concatenate([[self.B0], self.K * np.exp(H)])
        return B_final, history, initial_residual_norm

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------
    def price(self, S0, max_iters=10):
        """
        Price an American option at spot S0.

        Returns
        -------
        dict
            euro  : European component
            eep   : early exercise premium
            amer  : American price (euro + eep)
            runtime : elapsed seconds
            history : Newton iteration snapshots
            boundary : DataFrame with y, tau, B columns
            initial_residual_norm : initial residual norm before Newton
        """
        import pandas as pd

        t0 = time.perf_counter()
        B_final, history, initial_residual_norm = self.solve_boundary(max_iters=max_iters)

        # European price at (S0, T)
        pv = self._pv_divs(self.T)
        S_adj = max(S0 - pv, 1e-8)
        euro_price, _ = self._bs(S_adj, self.K, self.T)

        # EEP at S0: integrate f(S0, B(y_j), T - y_j^2, y_j) over j
        y = self.y
        f_vec = np.zeros(self.N)
        for j in range(self.N):
            tau_j = self.T - y[j] ** 2
            if tau_j < 1e-12:
                continue
            vt = self.sigma * np.sqrt(tau_j)
            d1 = (np.log(S_adj / B_final[j])
                  + (self.r - self.q + 0.5 * self.sigma ** 2) * tau_j) / vt
            d2 = d1 - vt
            f_vec[j] = 2 * y[j] * self.w * (
                self.q * S_adj * np.exp(-self.q * tau_j) * norm.cdf(self.w * d1)
                - self.r * self.K * np.exp(-self.r * tau_j) * norm.cdf(self.w * d2)
            )
        eep_val = max(0, self.A_rbf[-1, :] @ f_vec)

        runtime = time.perf_counter() - t0

        return {
            "euro": float(euro_price),
            "eep": float(eep_val),
            "amer": float(euro_price + eep_val),
            "runtime": runtime,
            "history": history,
            "boundary": pd.DataFrame(
                {"y": y, "tau": y ** 2, "B": B_final}),
            "initial_residual_norm": initial_residual_norm,
        }


# ===========================================================================
# Finite-difference Jacobian test
# ===========================================================================

def test_jacobian_fd(pricer, eps=1e-6):
    """
    Test analytic Jacobian against finite-difference approximation.

    Parameters
    ----------
    pricer : KimIntegralRBF
        Instance with boundary already initialized.
    eps : float
        Finite-difference step size.

    Returns
    -------
    dict
        max_error : maximum absolute error
        mean_error : mean absolute error
        fd_jac : finite-difference Jacobian
        analytic_jac : analytic Jacobian
    """
    # Get initial log-boundary (excluding knot 0)
    log_BK = pricer._initial_log_boundary()
    H = log_BK[1:]  # N-1 unknowns
    n = len(H)

    # Compute analytic residual and Jacobian
    res0, jac_analytic = pricer.get_residual_and_jac(H)

    # Compute finite-difference Jacobian
    jac_fd = np.zeros_like(jac_analytic)
    for j in range(n):
        H_plus = H.copy()
        H_plus[j] += eps
        res_plus, _ = pricer.get_residual_and_jac(H_plus)
        jac_fd[:, j] = (res_plus - res0) / eps

    # Compute errors
    diff = np.abs(jac_fd - jac_analytic)
    max_error = np.max(diff)
    mean_error = np.mean(diff)

    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "fd_jac": jac_fd,
        "analytic_jac": jac_analytic,
    }


# ===========================================================================
# Main: test cases and convergence plots
# ===========================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Test 1: American Put (no discrete dividends) ----
    print("=" * 60)
    print("American Put  (S=100, K=100, T=1, r=5%, q=5%, sigma=25%)")
    print("=" * 60)

    N=20
    max_iters = 4
    n_powers = 3

    pricer_put = KimIntegralRBF(
        K=100, T=1.0, r=0.05, q=0.05, sigma=0.25, w=-1, N=N, n_powers=n_powers)
    res_put = pricer_put.price(S0=100, max_iters=max_iters)
    print(f"  Euro : {res_put['euro']:.8f}")
    print(f"  EEP  : {res_put['eep']:.8f}")
    print(f"  Amer : {res_put['amer']:.8f}")
    print(f"  Init Residual: {res_put['initial_residual_norm']:.6e}")
    print(f"  Time : {res_put['runtime']:.4f}s")

    # ---- Test 2: American Call (continuous dividend) ----
    print()
    print("=" * 60)
    print("American Call (S=100, K=100, T=1, r=5%, q=5%, sigma=25%)")
    print("=" * 60)
    pricer_call = KimIntegralRBF(
        K=100, T=1.0, r=0.05, q=0.05, sigma=0.25, w=+1, N=N, n_powers=n_powers)
    res_call = pricer_call.price(S0=100, max_iters=max_iters)
    print(f"  Euro : {res_call['euro']:.8f}")
    print(f"  EEP  : {res_call['eep']:.8f}")
    print(f"  Amer : {res_call['amer']:.8f}")
    print(f"  Init Residual: {res_call['initial_residual_norm']:.6e}")
    print(f"  Time : {res_call['runtime']:.4f}s")

    # ---- Test 3: American Put with discrete dividends ----
    do_call = False
    print()
    print("=" * 60)
    print(f"American {'Call' if do_call else 'Put'}  (S=100, K=100, T=1, r=2%, q=2.0%, sigma=10%)")
    print("  Discrete dividends: D=2.0 at t=0.25, D=2.0 at t=0.75")
    print("=" * 60)
    pricer_div = KimIntegralRBF(
        K=100, T=1.0, r=0.02, q=0.02, sigma=0.10, w=+1 if do_call else -1, N=N, n_powers=n_powers,
        div_times=[0.25, 0.75], div_amounts=[2.0, 2.0])
    res_div = pricer_div.price(S0=100, max_iters=max_iters)
    print(f"  Euro : {res_div['euro']:.8f}")
    print(f"  EEP  : {res_div['eep']:.8f}")
    print(f"  Amer : {res_div['amer']:.8f}")
    print(f"  Init Residual: {res_div['initial_residual_norm']:.6e}")
    print(f"  Time : {res_div['runtime']:.4f}s")

    # ---- Test 4: Jacobian finite-difference validation ----
    print()
    print("=" * 60)
    print("Jacobian Finite-Difference Test")
    print("=" * 60)

    # Test on the put case
    print("\nTesting on American Put:")
    jac_test_put = test_jacobian_fd(pricer_put, eps=1e-6)
    print(f"  Max error  : {jac_test_put['max_error']:.6e}")
    print(f"  Mean error : {jac_test_put['mean_error']:.6e}")

    # Test on the call case
    print("\nTesting on American Call:")
    jac_test_call = test_jacobian_fd(pricer_call, eps=1e-6)
    print(f"  Max error  : {jac_test_call['max_error']:.6e}")
    print(f"  Mean error : {jac_test_call['mean_error']:.6e}")

    # Test on the discrete dividend case
    print(f"\nTesting on American {'Call' if do_call else 'Put'} with Dividends:")
    jac_test_div = test_jacobian_fd(pricer_div, eps=1e-6)
    print(f"  Max error  : {jac_test_div['max_error']:.6e}")
    print(f"  Mean error : {jac_test_div['mean_error']:.6e}")

    # ---- Test 5: Initial guess improvement comparison ----
    print()
    print("=" * 60)
    print("Initial Guess Improvement (Discrete Dividends)")
    print("=" * 60)

    # Compute initial residual with OLD method (simple linear interpolation)
    S_star_old = _ju_critical_price(pricer_div.K, pricer_div.T, pricer_div.r,
                                     pricer_div.q, pricer_div.sigma, pricer_div.w)
    log_BK_old = np.log(S_star_old / pricer_div.K) * pricer_div.y / pricer_div.y_max
    H_old = log_BK_old[1:]
    res_old, _ = pricer_div.get_residual_and_jac(H_old)
    residual_old = np.linalg.norm(res_old)

    # Compute initial residual with NEW method (with discrete div adjustment)
    residual_new = res_div['initial_residual_norm']

    print(f"\n  Old method (linear Ju):    {residual_old:.6e}")
    print(f"  New method (div-adjusted): {residual_new:.6e}")
    print(f"  Improvement factor:        {residual_old / residual_new:.2f}x")

    # ---- Test 6: More challenging discrete dividend case ----
    print()
    print("=" * 60)
    print("Challenging Case: Larger Dividends")
    print("American Put (S=100, K=100, T=1, r=5%, q=0%, sigma=20%)")
    print("  Discrete dividends: D=5.0 at t=0.25, D=5.0 at t=0.75")
    print("=" * 60)

    pricer_hard = KimIntegralRBF(
        K=100, T=1.0, r=0.05, q=0.0, sigma=0.20, w=-1, N=30, n_powers=2,
        div_times=[0.25, 0.75], div_amounts=[5.0, 5.0])
    res_hard = pricer_hard.price(S0=100, max_iters=max_iters)

    # Compare with old method
    S_star_hard_old = _ju_critical_price(pricer_hard.K, pricer_hard.T, pricer_hard.r,
                                          pricer_hard.q, pricer_hard.sigma, pricer_hard.w)
    log_BK_hard_old = np.log(S_star_hard_old / pricer_hard.K) * pricer_hard.y / pricer_hard.y_max
    H_hard_old = log_BK_hard_old[1:]
    res_hard_old, _ = pricer_hard.get_residual_and_jac(H_hard_old)
    residual_hard_old = np.linalg.norm(res_hard_old)
    residual_hard_new = res_hard['initial_residual_norm']

    print(f"\n  Old method (linear Ju):    {residual_hard_old:.6e}")
    print(f"  New method (div-adjusted): {residual_hard_new:.6e}")
    print(f"  Improvement factor:        {residual_hard_old / residual_hard_new:.2f}x")
    print(f"\n  American Put Price: {res_hard['amer']:.8f}")

    # ---- Convergence plots ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for label, (B, rmse) in res_put["history"].items():
        alpha = 0.8 if label==list(res_put["history"].keys())[-1] else 0.2
        ax.plot(pricer_put.y ** 2, B, label=f"{label} RMSE={rmse:.2e}", alpha=alpha)
    ax.set_title("Put Boundary Convergence")
    ax.set_xlabel(r"$\tau = y^2$")
    ax.set_ylabel("B(tau)")
    ax.invert_xaxis()
    ax.legend(ncol=1, fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for label, (B, rmse) in res_call["history"].items():
        alpha = 0.8 if label==list(res_call["history"].keys())[-1] else 0.2
        ax.plot(pricer_call.y ** 2, B, label=f"{label} RMSE={rmse:.2e}", alpha=alpha)
    ax.set_title("Call Boundary Convergence")
    ax.set_xlabel(r"$\tau = y^2$")
    ax.set_ylabel("B(tau)")
    ax.invert_xaxis()
    ax.legend(ncol=1, fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for label, (B, rmse) in res_div["history"].items():
        alpha = 0.8 if label==list(res_div["history"].keys())[-1] else 0.2
        ax.plot(pricer_div.y ** 2, B, label=f"{label} RMSE={rmse:.2e}", alpha=alpha)

    print(res_div["history"])
    
    for t_d, D in pricer_div.div_schedule:
        ax.axvline(pricer_div.T - t_d, color="gray", ls="--", lw=0.8)
    ax.set_title(f"{'Call' if do_call else 'Put'} + Discrete Div Boundary")
    ax.set_xlabel(r"$\tau = y^2$")
    ax.set_ylabel("B(tau)")
    ax.invert_xaxis()
    ax.legend(ncol=1, fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("kim_integral_rbf_convergence.png", dpi=150)
    print("\nSaved kim_integral_rbf_convergence.png")
