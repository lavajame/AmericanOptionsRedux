import numpy as np

class FiniteDifferenceAmerican:
    """
    Finite-difference American option pricer with support for:
    - Calls and puts
    - Continuous dividend yield (q)
    - Discrete cash dividends (optional, for later)
    - Early exercise boundary extraction (time-varying)

    This implementation uses a Crank-Nicolson scheme with optional
    Rannacher smoothing (two initial implicit steps) and PSOR
    to enforce early exercise.
    """
    def __init__(
        self,
        S0,
        K,
        r,
        T,
        sigma,
        q=0.0,
        option_type="call",
        is_american=True,
        Smax=None,
        M=400,
        N=2000,
        omega=1.2,
        tol=1e-8,
        max_iter=10000,
        rannacher_steps=2,
        discrete_dividends=None,
    ):
        self.S0 = float(S0)
        self.K = float(K)
        self.r = float(r)
        self.T = float(T)
        self.sigma = float(sigma)
        self.q = float(q)
        self.option_type = option_type.lower()
        self.is_american = bool(is_american)
        self.M = int(M)
        self.N = int(N)
        self.omega = float(omega)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.rannacher_steps = int(rannacher_steps)
        self.discrete_dividends = discrete_dividends or []

        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")

        if Smax is None:
            # Heuristic Smax. Increase for high vol/long maturities.
            mult = 4.0 if self.option_type == "call" else 3.0
            Smax = max(self.K, self.S0) * mult
        self.Smax = float(Smax)

        # Sort discrete dividends by time
        self.discrete_dividends = sorted(
            [(float(t), float(D)) for t, D in self.discrete_dividends],
            key=lambda x: x[0],
        )

    def _payoff(self, S):
        if self.option_type == "call":
            return np.maximum(S - self.K, 0.0)
        return np.maximum(self.K - S, 0.0)

    def _boundary_conditions(self, S_grid, tau):
        # tau = time to maturity at current step
        if self.option_type == "call":
            # With continuous dividends q, large-S behavior
            # V ~ S e^{-q tau} - K e^{-r tau}
            left = 0.0
            right = self.Smax * np.exp(-self.q * tau) - self.K * np.exp(-self.r * tau)
        else:
            # Put
            left = self.K * np.exp(-self.r * tau)
            right = 0.0
        return left, right

    def _setup_grid(self):
        S = np.linspace(0.0, self.Smax, self.M + 1)
        dS = S[1] - S[0]
        return S, dS

    def _setup_coeffs(self, S, dt):
        # Coefficients for Crank-Nicolson: A * V^{n+1} = B * V^{n} + b
        i = np.arange(1, self.M)
        Si = S[i]

        a = 0.25 * dt * (self.sigma**2 * (Si / (S[1] - S[0]))**2 - (self.r - self.q) * Si / (S[1] - S[0]))
        b = -0.5 * dt * (self.sigma**2 * (Si / (S[1] - S[0]))**2 + self.r)
        c = 0.25 * dt * (self.sigma**2 * (Si / (S[1] - S[0]))**2 + (self.r - self.q) * Si / (S[1] - S[0]))

        # A matrix diagonals (implicit)
        A_lower = -a
        A_diag = 1 - b
        A_upper = -c

        # B matrix diagonals (explicit)
        B_lower = a
        B_diag = 1 + b
        B_upper = c

        return A_lower, A_diag, A_upper, B_lower, B_diag, B_upper

    def _psor(self, A_lower, A_diag, A_upper, rhs, intrinsic, initial):
        # Projected SOR for American constraints
        V = initial.copy()
        for _ in range(self.max_iter):
            V_old = V.copy()
            for j in range(len(V)):
                left = A_lower[j] * V[j - 1] if j > 0 else 0.0
                right = A_upper[j] * V[j + 1] if j < len(V) - 1 else 0.0
                V_new = (rhs[j] - left - right) / A_diag[j]
                V[j] = max(intrinsic[j], V[j] + self.omega * (V_new - V[j]))

            if np.max(np.abs(V - V_old)) < self.tol:
                break
        return V

    def _apply_dividend(self, S, V, D, enforce_intrinsic: bool = True):
        # Cash dividend: S -> S - D
        # V(S, t-) = V(S - D, t+)
        shifted = np.maximum(S - D, 0.0)
        V_new = np.interp(shifted, S, V)
        if enforce_intrinsic and self.is_american:
            V_new = np.maximum(V_new, self._payoff(S))
        return V_new

    def price(self, return_boundary=False):
        if self.T <= 0:
            return self._payoff(np.array([self.S0]))[0] if not return_boundary else (
                self._payoff(np.array([self.S0]))[0], []
            )

        S, dS = self._setup_grid()
        dt = self.T / self.N

        # Initial condition at maturity
        V = self._payoff(S)

        # Map dividend times to time steps (backward in time)
        div_steps = {}
        for t_div, D in self.discrete_dividends:
            if 0.0 < t_div <= self.T:
                step = int(round(t_div / dt))
                div_steps[step] = D

        # Boundary tracking
        boundary = []

        for n in range(self.N):
            tau = self.T - n * dt

            left_bc, right_bc = self._boundary_conditions(S, tau)
            V[0] = left_bc
            V[-1] = right_bc

            # Build coefficients (CN or implicit for Rannacher steps)
            if n < self.rannacher_steps:
                # Fully implicit: B = I
                A_lower, A_diag, A_upper, _, _, _ = self._setup_coeffs(S, dt)
                rhs = V[1:-1].copy()
            else:
                A_lower, A_diag, A_upper, B_lower, B_diag, B_upper = self._setup_coeffs(S, dt)
                rhs = (
                    B_lower * V[:-2]
                    + B_diag * V[1:-1]
                    + B_upper * V[2:]
                )

            # Boundary contributions
            rhs[0] -= A_lower[0] * V[0]
            rhs[-1] -= A_upper[-1] * V[-1]

            # Intrinsic and solve
            intrinsic = self._payoff(S[1:-1])

            V_cont_inner = None
            if self.is_american and return_boundary:
                V_cont_inner = self._solve_tridiagonal(A_lower, A_diag, A_upper, rhs)

            if self.is_american:
                V_inner = self._psor(A_lower, A_diag, A_upper, rhs, intrinsic, V[1:-1])
            else:
                V_inner = self._solve_tridiagonal(A_lower, A_diag, A_upper, rhs)

            V[1:-1] = V_inner
            if self.is_american:
                V = np.maximum(V, self._payoff(S))

            # Apply discrete dividend if needed at this step
            step_index = self.N - n
            if step_index in div_steps:
                V = self._apply_dividend(S, V, div_steps[step_index])

            V_cont_full = None
            if V_cont_inner is not None:
                V_cont_full = V.copy()
                V_cont_full[1:-1] = V_cont_inner
                if step_index in div_steps:
                    V_cont_full = self._apply_dividend(S, V_cont_full, div_steps[step_index], enforce_intrinsic=False)

            # Extract early exercise boundary
            if return_boundary and self.is_american:
                V_for_boundary = V_cont_full if V_cont_full is not None else V
                boundary.append((self.T - (n + 1) * dt, self._extract_boundary(S, V_for_boundary)))

        price = np.interp(self.S0, S, V)
        if return_boundary:
            return price, boundary
        return price

    def _solve_tridiagonal(self, a, b, c, d):
        # Thomas algorithm
        n = len(d)
        c_ = np.zeros(n)
        d_ = np.zeros(n)

        c_[0] = c[0] / b[0]
        d_[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_[i - 1]
            c_[i] = c[i] / denom if i < n - 1 else 0.0
            d_[i] = (d[i] - a[i] * d_[i - 1]) / denom

        x = np.zeros(n)
        x[-1] = d_[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_[i] - c_[i] * x[i + 1]
        return x

    def _extract_boundary(self, S, V):
        payoff = self._payoff(S)
        # Identify continuation region robustly: V should exceed payoff by a tolerance
        tol = 1e-8
        continuation = V > payoff + tol

        if self.option_type == "put":
            # Continuation is at high S; find first continuation index
            cont_idx = np.where(continuation)[0]
            if len(cont_idx) == 0:
                return None
            i = cont_idx[0]
            if i <= 0:
                return None
            i0, i1 = i - 1, i
        else:
            # Call: continuation is at low S; find last continuation index
            cont_idx = np.where(continuation)[0]
            if len(cont_idx) == 0:
                return None
            i = cont_idx[-1]
            if i >= len(S) - 1:
                return None
            i0, i1 = i, i + 1

        y0 = V[i0] - payoff[i0]
        y1 = V[i1] - payoff[i1]
        if abs(y1 - y0) < 1e-12:
            return float(S[i0])
        t = -y0 / (y1 - y0)
        return float(S[i0] + t * (S[i1] - S[i0]))
