import numpy as np
from math import exp, log, sqrt
from scipy.stats import norm
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt


def european_call_bs(S0, K, r, q, sigma, T):
    if T <= 0.0:
        return max(S0 - K, 0.0)
    vol_sqrt_T = sigma * sqrt(T)
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return S0 * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def european_price_w(S0, K, r, q, sigma, T, w):
    call = european_call_bs(S0, K, r, q, sigma, T)
    if w == 1:
        return call
    return call - S0 * exp(-q * T) + K * exp(-r * T)


def build_bspline_knots(T, n_basis, degree=3, knot_strategy="uniform", power=2.0):
    n_internal = n_basis - degree - 1
    if n_internal < 0:
        raise ValueError("n_basis must be >= degree + 1")
    if n_internal == 0:
        internal = np.array([])
    else:
        if knot_strategy == "uniform":
            internal = np.linspace(0.0, T, n_internal + 2)[1:-1]
        elif knot_strategy == "power":
            u = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
            internal = T * (u ** power)
        else:
            raise ValueError(f"Unknown knot_strategy: {knot_strategy}")
    knots = np.concatenate([
        np.zeros(degree + 1),
        internal,
        np.full(degree + 1, T),
    ])
    return knots


def build_bspline_basis_matrix(tau, knots, degree=3, add_sqrt=False):
    tau = np.asarray(tau)
    n_basis = len(knots) - degree - 1
    basis = np.zeros((tau.size, n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        spline = BSpline(knots, c, degree, extrapolate=False)
        vals = spline(tau)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        basis[:, i] = vals

    if add_sqrt:
        sqrt_tau = np.sqrt(np.maximum(tau, 0.0))
        basis = np.column_stack([basis, sqrt_tau])

    return basis


def boundary_from_bspline(times, T, coeffs, knots, degree=3, basis_matrix=None):
    if basis_matrix is None:
        tau = np.maximum(T - times, 0.0)
        basis_matrix = build_bspline_basis_matrix(tau, knots, degree)
    return basis_matrix @ coeffs


def _build_interp_cache(times, u_grid):
    times = np.asarray(times)
    u_grid = np.asarray(u_grid)
    idx_hi = np.searchsorted(times, u_grid, side="left")
    idx_hi = np.clip(idx_hi, 1, len(times) - 1)
    idx_lo = idx_hi - 1
    t_lo = times[idx_lo]
    t_hi = times[idx_hi]
    denom = np.maximum(t_hi - t_lo, 1e-14)
    w_hi = (u_grid - t_lo) / denom
    return idx_lo, idx_hi, w_hi


def kim_eep_from_t(Sj, j, times, B_grid, r, q, sigma, K, w, n_int_steps=60, interp_cache=None):
    T = times[-1]
    t0 = times[j]
    if t0 >= T:
        return 0.0
    u_grid = np.linspace(t0, T, n_int_steps + 1)
    du = (T - t0) / n_int_steps

    if interp_cache is None:
        idx_lo, idx_hi, w_hi = _build_interp_cache(times, u_grid)
    else:
        idx_lo, idx_hi, w_hi = interp_cache

    Bt = (1.0 - w_hi) * B_grid[idx_lo] + w_hi * B_grid[idx_hi]
    tau = u_grid - t0
    mask = tau > 0.0

    eep_vals = np.zeros_like(u_grid)
    if np.any(mask):
        tau_m = tau[mask]
        Bt_m = Bt[mask]
        valid = (Bt_m > 0.0) & np.isfinite(Bt_m)
        if np.any(valid):
            tau_v = tau_m[valid]
            Bt_v = Bt_m[valid]
            vol_sqrt_tau = sigma * np.sqrt(tau_v)
            d1B = (np.log(Sj / Bt_v) + (r - q + 0.5 * sigma**2) * tau_v) / vol_sqrt_tau
            d2B = d1B - vol_sqrt_tau
            if w == 1:
                ES_indicator = Sj * np.exp((r - q) * tau_v) * norm.cdf(d1B)
                prob_indicator = norm.cdf(d2B)
                vals = np.exp(-r * tau_v) * (q * ES_indicator - r * K * prob_indicator)
            else:
                ES_indicator = Sj * np.exp((r - q) * tau_v) * norm.cdf(-d1B)
                prob_indicator = norm.cdf(-d2B)
                vals = np.exp(-r * tau_v) * (r * K * prob_indicator - q * ES_indicator)
            idx = np.where(mask)[0][valid]
            eep_vals[idx] = vals

    return 0.5 * du * (eep_vals[0] + 2.0 * eep_vals[1:-1].sum() + eep_vals[-1])


# -------- piecewise polynomial boundary in sqrt(time to next event) --------

def boundary_piecewise(times, event_times, coeffs_list):
    """
    Piecewise boundary:
    For t in [event_times[i], event_times[i+1]], define tau = event_times[i+1] - t.
    B(t) = sum_k c_{i,k} * (sqrt(tau))^k
    """
    times = np.asarray(times)
    B = np.zeros_like(times, dtype=float)

    for seg in range(len(event_times) - 1):
        t_left = event_times[seg]
        t_right = event_times[seg + 1]
        mask = (times >= t_left) & (times <= t_right)
        tau = np.maximum(t_right - times[mask], 0.0)
        x = np.sqrt(tau)
        c = coeffs_list[seg]
        powers = np.vstack([x**k for k in range(len(c))])
        B[mask] = (c[:, None] * powers).sum(axis=0)

    # enforce B(T) = K (caller should do this if desired)
    return B


def _segment_index_for_time(t, event_times):
    for i in range(len(event_times) - 1):
        if event_times[i] <= t <= event_times[i + 1]:
            return i
    return len(event_times) - 2


def residuals_piecewise(coeffs_list, times, event_times, K, r, q, sigma, w,
                        n_int_steps_eep=60, n_residuals_per_seg=10):
    B = boundary_piecewise(times, event_times, coeffs_list)
    B[-1] = K

    res = []
    # choose residual time indices per segment (exclude endpoints)
    # cache u-grids and interpolation weights per residual time index
    u_cache = {}

    for seg in range(len(event_times) - 1):
        t_left = event_times[seg]
        t_right = event_times[seg + 1]
        idxs = np.where((times > t_left) & (times < t_right))[0]
        if idxs.size == 0:
            continue
        pick = np.linspace(0, idxs.size - 1, n_residuals_per_seg, dtype=int)
        for j in idxs[pick]:
            if j not in u_cache:
                u_grid = np.linspace(times[j], times[-1], n_int_steps_eep + 1)
                u_cache[j] = (u_grid, (u_grid[-1] - u_grid[0]) / n_int_steps_eep, _build_interp_cache(times, u_grid))
            Bj = max(B[j], 1e-10)
            CE = european_price_w(Bj, K, r, q, sigma, times[-1] - times[j], w)
            _, _, interp_cache = u_cache[j]
            eep = kim_eep_from_t(Bj, j, times, B, r, q, sigma, K, w,
                                 n_int_steps=n_int_steps_eep, interp_cache=interp_cache)
            CA = CE + eep
            if w == 1:
                res.append(CA - (Bj - K))
            else:
                res.append(CA - (K - Bj))

    return np.asarray(res)


def jacobian_piecewise(coeffs_list, times, event_times, K, r, q, sigma, w,
                        n_int_steps_eep=60, n_residuals_per_seg=10):
    """
    Approximate Jacobian using dEEP/dB_j only (same approximation as Kim-style GN).
    """
    B = boundary_piecewise(times, event_times, coeffs_list)
    B[-1] = K

    # flatten parameters
    param_sizes = [len(c) for c in coeffs_list]
    total_params = sum(param_sizes)

    rows = []
    # precompute segment param offsets
    offsets = np.cumsum([0] + param_sizes[:-1])

    u_cache = {}

    for seg in range(len(event_times) - 1):
        t_left = event_times[seg]
        t_right = event_times[seg + 1]
        idxs = np.where((times > t_left) & (times < t_right))[0]
        if idxs.size == 0:
            continue
        pick = np.linspace(0, idxs.size - 1, n_residuals_per_seg, dtype=int)
        for j in idxs[pick]:
            if j not in u_cache:
                u_grid = np.linspace(times[j], times[-1], n_int_steps_eep + 1)
                u_cache[j] = (u_grid, (u_grid[-1] - u_grid[0]) / n_int_steps_eep, _build_interp_cache(times, u_grid))
            Bj = max(B[j], 1e-10)
            tau_j = times[-1] - times[j]
            if tau_j <= 0:
                continue
            vol = sigma * np.sqrt(tau_j)
            d1 = (np.log(Bj / K) + (r - q + 0.5 * sigma**2) * tau_j) / vol
            if w == 1:
                dCE_dB = np.exp(-q * tau_j) * norm.cdf(d1)
            else:
                dCE_dB = np.exp(-q * tau_j) * (norm.cdf(d1) - 1.0)

            u_grid, du, interp_cache = u_cache[j]
            idx_lo, idx_hi, w_hi = interp_cache
            Bt = (1.0 - w_hi) * B[idx_lo] + w_hi * B[idx_hi]
            tau = u_grid - times[j]
            mask = tau > 0.0
            dEEP_vals = np.zeros_like(u_grid)
            if np.any(mask):
                tau_m = tau[mask]
                Bt_m = Bt[mask]
                valid = (Bt_m > 0.0) & np.isfinite(Bt_m)
                if np.any(valid):
                    tau_v = tau_m[valid]
                    Bt_v = Bt_m[valid]
                    vol_u = sigma * np.sqrt(tau_v)
                    d1u = (np.log(Bj / Bt_v) + (r - q + 0.5 * sigma**2) * tau_v) / vol_u
                    d2u = d1u - vol_u
                    if w == 1:
                        term1 = q * np.exp((r - q) * tau_v) * norm.cdf(d1u)
                        term2 = r * K * norm.pdf(d2u) / (Bj * sigma * np.sqrt(tau_v))
                        vals = np.exp(-r * tau_v) * (term1 - term2)
                    else:
                        term1 = q * np.exp((r - q) * tau_v) * norm.cdf(-d1u)
                        term2 = r * K * norm.pdf(d2u) / (Bj * sigma * np.sqrt(tau_v))
                        vals = np.exp(-r * tau_v) * (-term1 + term2)
                    idx = np.where(mask)[0][valid]
                    dEEP_vals[idx] = vals

            dEEP_dB = 0.5 * du * (dEEP_vals[0] + 2.0 * dEEP_vals[1:-1].sum() + dEEP_vals[-1])
            dr_dBj = dCE_dB + dEEP_dB - w

            row = np.zeros(total_params)
            seg_idx = _segment_index_for_time(times[j], event_times)
            c = coeffs_list[seg_idx]
            t_right = event_times[seg_idx + 1]
            xj = np.sqrt(max(t_right - times[j], 0.0))
            offset = offsets[seg_idx]
            for k in range(len(c)):
                row[offset + k] = dr_dBj * (xj ** k)
            rows.append(row)

    if rows:
        return np.vstack(rows)
    return np.zeros((0, total_params))


def gauss_newton_piecewise(coeffs_list, times, event_times, K, r, q, sigma, w,
                           n_int_steps_eep=60, n_residuals_per_seg=10,
                           lam=1e-3, tol=1e-6, max_iter=12):
    coeffs = [c.copy() for c in coeffs_list]

    for _ in range(max_iter):
        r_vec = residuals_piecewise(coeffs, times, event_times, K, r, q, sigma, w,
                                    n_int_steps_eep=n_int_steps_eep,
                                    n_residuals_per_seg=n_residuals_per_seg)
        J = jacobian_piecewise(coeffs, times, event_times, K, r, q, sigma, w,
                               n_int_steps_eep=n_int_steps_eep,
                               n_residuals_per_seg=n_residuals_per_seg)

        if J.size == 0:
            break

        JTJ = J.T @ J + lam * np.eye(J.shape[1])
        JTr = J.T @ r_vec

        try:
            delta = np.linalg.solve(JTJ, -JTr)
        except np.linalg.LinAlgError:
            break

        # update coefficients
        idx = 0
        for seg in range(len(coeffs)):
            p = len(coeffs[seg])
            coeffs[seg] = coeffs[seg] + delta[idx:idx + p]
            idx += p

        if np.linalg.norm(delta) < tol * max(1.0, np.linalg.norm(np.concatenate(coeffs))):
            break

    return coeffs


def residuals_bspline(coeffs, basis_matrix, times, K, r, q, sigma, w,
                      n_int_steps_eep=60, n_residuals=12):
    B = basis_matrix @ coeffs
    B[-1] = K

    idxs = np.linspace(1, len(times) - 2, n_residuals, dtype=int)
    res = []

    u_cache = {}
    for j in idxs:
        if j not in u_cache:
            u_grid = np.linspace(times[j], times[-1], n_int_steps_eep + 1)
            u_cache[j] = (u_grid, (u_grid[-1] - u_grid[0]) / n_int_steps_eep, _build_interp_cache(times, u_grid))
        Bj = max(B[j], 1e-10)
        CE = european_price_w(Bj, K, r, q, sigma, times[-1] - times[j], w)
        _, _, interp_cache = u_cache[j]
        eep = kim_eep_from_t(Bj, j, times, B, r, q, sigma, K, w,
                             n_int_steps=n_int_steps_eep, interp_cache=interp_cache)
        CA = CE + eep
        if w == 1:
            res.append(CA - (Bj - K))
        else:
            res.append(CA - (K - Bj))

    return np.asarray(res)


def jacobian_bspline(coeffs, basis_matrix, times, K, r, q, sigma, w,
                     n_int_steps_eep=60, n_residuals=12):
    B = basis_matrix @ coeffs
    B[-1] = K

    idxs = np.linspace(1, len(times) - 2, n_residuals, dtype=int)
    rows = []
    u_cache = {}

    for j in idxs:
        if j not in u_cache:
            u_grid = np.linspace(times[j], times[-1], n_int_steps_eep + 1)
            u_cache[j] = (u_grid, (u_grid[-1] - u_grid[0]) / n_int_steps_eep, _build_interp_cache(times, u_grid))
        Bj = max(B[j], 1e-10)
        tau_j = times[-1] - times[j]
        if tau_j <= 0:
            continue
        vol = sigma * np.sqrt(tau_j)
        d1 = (np.log(Bj / K) + (r - q + 0.5 * sigma**2) * tau_j) / vol
        if w == 1:
            dCE_dB = np.exp(-q * tau_j) * norm.cdf(d1)
        else:
            dCE_dB = np.exp(-q * tau_j) * (norm.cdf(d1) - 1.0)

        u_grid, du, interp_cache = u_cache[j]
        idx_lo, idx_hi, w_hi = interp_cache
        Bt = (1.0 - w_hi) * B[idx_lo] + w_hi * B[idx_hi]
        tau = u_grid - times[j]
        mask = tau > 0.0
        dEEP_vals = np.zeros_like(u_grid)
        if np.any(mask):
            tau_m = tau[mask]
            Bt_m = Bt[mask]
            valid = (Bt_m > 0.0) & np.isfinite(Bt_m)
            if np.any(valid):
                tau_v = tau_m[valid]
                Bt_v = Bt_m[valid]
                vol_u = sigma * np.sqrt(tau_v)
                d1u = (np.log(Bj / Bt_v) + (r - q + 0.5 * sigma**2) * tau_v) / vol_u
                d2u = d1u - vol_u
                if w == 1:
                    term1 = q * np.exp((r - q) * tau_v) * norm.cdf(d1u)
                    term2 = r * K * norm.pdf(d2u) / (Bj * sigma * np.sqrt(tau_v))
                    vals = np.exp(-r * tau_v) * (term1 - term2)
                else:
                    term1 = q * np.exp((r - q) * tau_v) * norm.cdf(-d1u)
                    term2 = r * K * norm.pdf(d2u) / (Bj * sigma * np.sqrt(tau_v))
                    vals = np.exp(-r * tau_v) * (-term1 + term2)
                idx = np.where(mask)[0][valid]
                dEEP_vals[idx] = vals

        dEEP_dB = 0.5 * du * (dEEP_vals[0] + 2.0 * dEEP_vals[1:-1].sum() + dEEP_vals[-1])
        dr_dBj = dCE_dB + dEEP_dB - w

        row = dr_dBj * basis_matrix[j, :]
        rows.append(row)

    if rows:
        return np.vstack(rows)
    return np.zeros((0, basis_matrix.shape[1]))


def gauss_newton_bspline(coeffs, basis_matrix, times, K, r, q, sigma, w,
                         n_int_steps_eep=60, n_residuals=12,
                         lam=1e-3, tol=1e-6, max_iter=12):
    c = coeffs.copy()
    for _ in range(max_iter):
        r_vec = residuals_bspline(c, basis_matrix, times, K, r, q, sigma, w,
                                  n_int_steps_eep=n_int_steps_eep,
                                  n_residuals=n_residuals)
        J = jacobian_bspline(c, basis_matrix, times, K, r, q, sigma, w,
                             n_int_steps_eep=n_int_steps_eep,
                             n_residuals=n_residuals)

        if J.size == 0:
            break

        JTJ = J.T @ J + lam * np.eye(J.shape[1])
        JTr = J.T @ r_vec

        try:
            delta = np.linalg.solve(JTJ, -JTr)
        except np.linalg.LinAlgError:
            break

        c = c + delta
        if np.linalg.norm(delta) < tol * max(1.0, np.linalg.norm(c)):
            break

    return c


def american_call_binomial_with_boundary(S0, K, r, q, sigma, T, N):
    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    disc = exp(-r * dt)
    p = (exp((r - q) * dt) - d) / (u - d)

    S = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        for i in range(j + 1):
            S[j, i] = S0 * (u ** i) * (d ** (j - i))

    V = np.zeros_like(S)
    V[N, :] = np.maximum(S[N, :] - K, 0.0)

    B = np.full(N + 1, np.inf)

    for j in range(N - 1, -1, -1):
        exercise_indices = []
        for i in range(j + 1):
            continuation = disc * (p * V[j + 1, i + 1] + (1.0 - p) * V[j + 1, i])
            exercise = max(S[j, i] - K, 0.0)
            V[j, i] = max(continuation, exercise)
            if exercise > continuation + 1e-10:
                exercise_indices.append(i)
        if exercise_indices:
            B[j] = min(S[j, i] for i in exercise_indices)
        else:
            B[j] = np.inf

    times = np.linspace(0.0, T, N + 1)
    B[-1] = K

    finite_mask = np.isfinite(B)
    if np.any(finite_mask):
        first_finite_idx = np.argmax(finite_mask)
        first_finite_val = B[first_finite_idx]
        B[:first_finite_idx] = first_finite_val
    else:
        B[:] = np.inf

    for j in range(len(B) - 1):
        if B[j + 1] > B[j]:
            B[j + 1] = B[j]

    return V[0, 0], times, B


def price_american_from_boundary(S0, K, r, q, sigma, T, times, B, w,
                                 n_int_steps_eep=80):
    CE0 = european_call_bs(S0, K, r, q, sigma, T)
    u_grid = np.linspace(times[0], times[-1], n_int_steps_eep + 1)
    interp_cache = _build_interp_cache(times, u_grid)
    eep0 = kim_eep_from_t(S0, 0, times, B, r, q, sigma, K, w,
                          n_int_steps=n_int_steps_eep, interp_cache=interp_cache)
    if w == 1:
        return CE0 + eep0
    return CE0 - S0 * np.exp(-q * T) + K * np.exp(-r * T) + eep0


def main():
    S0 = 125.0
    K = 100.0
    r = 0.01
    q = 0.06
    sigma = 0.15
    T = 1.0
    w = 1

    # events (e.g., dividend dates) – last event must be T
    event_times = [0.0, 0.6, T]

    # grid for evaluation
    N_time = 120
    times = np.linspace(0.0, T, N_time + 1)

    # polynomial degree per segment (low order)
    deg_per_seg = [5, 5]

    coeffs0 = []
    for seg, deg in enumerate(deg_per_seg):
        p = deg + 1
        t_right = event_times[seg + 1]
        tau0 = max(t_right - event_times[seg], 1e-8)
        x0 = np.sqrt(tau0)
        c0 = np.zeros(p)
        c0[0] = K
        if p > 1:
            c0[1] = (S0 - K) / max(x0, 1e-8)
        coeffs0.append(c0)

    n_residuals_per_seg = max(2, max(deg_per_seg) + 2)

    coeffs_opt = gauss_newton_piecewise(
        coeffs0, times, event_times, K, r, q, sigma, w,
        n_int_steps_eep=60, n_residuals_per_seg=n_residuals_per_seg,
        lam=1e-3, tol=1e-6, max_iter=12
    )

    B = boundary_piecewise(times, event_times, coeffs_opt)
    B[-1] = K

    CA = price_american_from_boundary(S0, K, r, q, sigma, T, times, B, w,
                                      n_int_steps_eep=80)
    CE = european_call_bs(S0, K, r, q, sigma, T)
    intrinsic = max(S0 - K, 0.0)

    bin_american, _, _ = american_call_binomial_with_boundary(
        S0, K, r, q, sigma, T, N=1000
    )

    print("Piecewise poly boundary (sqrt time to next event)")
    print("Event times:", event_times)
    print("Coefficients per segment:")
    for i, c in enumerate(coeffs_opt):
        print(f"  seg {i}: {c}")
    print(f"Intrinsic value          : {intrinsic: .6f}")
    print(f"European BS price        : {CE: .6f}")
    print(f"American (Kim, poly B)   : {CA: .6f}")
    print(f"American (Binomial)      : {bin_american: .6f}")
    print("-" * 60)
    print(f"American - Intrinsic     : {CA - intrinsic: .6e}")
    print(f"American - Euro          : {CA - CE: .6e}")

    plt.figure(figsize=(8, 5))
    plt.plot(times, B, label="Piecewise B(t)", linewidth=2)
    for ev in event_times:
        plt.axvline(ev, color="gray", linestyle="--", alpha=0.4)
    plt.axhline(K, color="black", linestyle="--", label="Strike K")
    plt.xlabel("Time t")
    plt.ylabel("Boundary B(t)")
    plt.title("Piecewise Polynomial Boundary in sqrt(time to next event)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
