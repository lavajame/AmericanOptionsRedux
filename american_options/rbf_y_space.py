"""
RBF-based American option pricer in y-space (y = sqrt(T-t)) [EXPERIMENTAL - IN DEVELOPMENT].

**STATUS**: This module is under development. The RBF interpolation and space transformation
framework is in place, but the boundary equation transformation needs debugging.

Uses Gaussian RBF fitted at N knots in y-space with iterative boundary refinement.
The shape parameter is set such that exp(-shape*width) = 0.01, where width = y_max = sqrt(T).

The Kim integral is transformed to y-space:
  τ = T - t  →  y = sqrt(τ)  →  dy = 1/(2*sqrt(τ)) dτ = 1/(2y) dτ
  
So dτ = 2y dy, and substitution accounts for the Jacobian in integrands.

**TODO**:
- Debug boundary update equation - currently producing incorrect boundaries
- Verify integral transformation from τ-space to y-space
- Add proper convergence criteria and damping
- Compare against FDM and Kim-Jang reference implementations
- Implement call option support via put-call duality
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.linalg import solve

from .gauss_kronrod import GaussKronrodRule, GKRule


class GaussianRBFInterpolator:
    """
    Gaussian RBF interpolator in 1D.
    
    Evaluates as:
        f(y) = sum_i c_i * exp(-shape * (y - y_i)^2)
    """
    
    def __init__(self, y_knots, coeffs, shape):
        """
        Args:
            y_knots: array of knot locations
            coeffs: array of RBF coefficients
            shape: shape parameter (controls RBF width)
        """
        self.y_knots = np.asarray(y_knots, dtype=float)
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.shape = float(shape)
    
    def __call__(self, y):
        """Evaluate RBF at point(s) y."""
        y = np.asarray(y, dtype=float)
        scalar_input = y.ndim == 0
        y = np.atleast_1d(y)
        
        # Gaussian kernel: exp(-shape * (y - y_i)^2)
        distances_sq = (y[:, np.newaxis] - self.y_knots[np.newaxis, :]) ** 2
        kernels = np.exp(-self.shape * distances_sq)  # shape: (len(y), len(y_knots))
        
        result = np.dot(kernels, self.coeffs)
        
        return float(result) if scalar_input else result


class RBFYSpaceAmerican:
    """
    Iterative American option pricer using RBF interpolation in y-space.
    
    Key features:
    - Transform τ = T-t to y = sqrt(τ)
    - Fit Gaussian RBF at N knots in y-space
    - Shape parameter: exp(-shape*sqrt(T)) = 0.01
    - Integrate using RBF interpolator over [0, sqrt(T)]
    - Iterate until convergence
    """
    
    def __init__(
        self,
        K,
        r,
        T,
        sigma,
        alpha=0.0,
        option_type="put",
        gk_rule=GKRule.BALANCED,
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
        
        # y-space max: y_max = sqrt(T)
        self.y_max = np.sqrt(self.T)
        
        # Shape parameter such that exp(-shape * y_max) = 0.01
        # shape = -ln(0.01) / y_max = ln(100) / sqrt(T)
        self.shape = np.log(100.0) / self.y_max if self.y_max > 0 else 1.0
    
    def _d1(self, S, tau, B, r=None, alpha=None):
        """d1 from Black-Scholes."""
        if tau <= 0:
            return np.inf if S > B else -np.inf
        if not np.isfinite(B) or B <= 0:
            return -np.inf
        if S <= 0:
            return -np.inf
        r = self.r if r is None else r
        alpha = self.alpha if alpha is None else alpha
        return (np.log(S / B) + (r - alpha + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
    
    def _d2(self, S, tau, B, r=None, alpha=None):
        """d2 from Black-Scholes."""
        return self._d1(S, tau, B, r=r, alpha=alpha) - self.sigma * np.sqrt(tau)
    
    def _euro_put(self, S, tau, r=None, alpha=None):
        """European put value."""
        if tau <= 0:
            return max(self.K - S, 0.0)
        r = self.r if r is None else r
        alpha = self.alpha if alpha is None else alpha
        d1 = self._d1(S, tau, self.K, r=r, alpha=alpha)
        d2 = d1 - self.sigma * np.sqrt(tau)
        return self.K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-alpha * tau) * norm.cdf(-d1)
    
    def _euro_call(self, S, tau, r=None, alpha=None):
        """European call value."""
        if tau <= 0:
            return max(S - self.K, 0.0)
        r = self.r if r is None else r
        alpha = self.alpha if alpha is None else alpha
        d1 = self._d1(S, tau, self.K, r=r, alpha=alpha)
        d2 = d1 - self.sigma * np.sqrt(tau)
        return S * np.exp(-alpha * tau) * norm.cdf(d1) - self.K * np.exp(-r * tau) * norm.cdf(d2)
    
    def _boundary_initial(self):
        """Initial boundary guess (put)."""
        if self.alpha > 0:
            return min(self.K, (self.r / self.alpha) * self.K)
        return self.K
    
    def _tau_to_y(self, tau):
        """Convert time-to-maturity τ to y = sqrt(τ)."""
        return np.sqrt(np.maximum(tau, 0.0))
    
    def _y_to_tau(self, y):
        """Convert y back to τ = y^2."""
        return y ** 2
    
    def _build_rbf_system(self, y_knots, B_vals_at_knots):
        """
        Build and solve RBF interpolation system.
        
        System: [φ_ij] * c = B, where φ_ij = exp(-shape * (y_i - y_j)^2)
        
        Args:
            y_knots: array of knot locations in y-space
            B_vals_at_knots: boundary values at knots
        
        Returns:
            coeffs: RBF coefficients
        """
        n = len(y_knots)
        # Build RBF matrix
        distances_sq = (y_knots[:, np.newaxis] - y_knots[np.newaxis, :]) ** 2
        A = np.exp(-self.shape * distances_sq)
        
        # Regularize (add small diagonal term for stability)
        A += 1e-10 * np.eye(n)
        
        # Solve for coefficients
        try:
            coeffs = solve(A, B_vals_at_knots)
        except np.linalg.LinAlgError:
            # Fallback: use least-squares
            coeffs, _, _, _ = np.linalg.lstsq(A, B_vals_at_knots, rcond=None)
        
        return coeffs
    
    def _integral_singular_y_space(self, y, B_y, rbf_interp, use_d1=False):
        """
        Compute singular integral in y-space.
        
        Original: ∫_0^τ exp(-r*s - 0.5*d2^2) / (σ*sqrt(2π*s)) ds
        
        With substitution s = u^2 (in τ space):
          ∫_0^sqrt(τ) 2u * ... du
        
        In y-space where y = sqrt(τ):
          τ = y^2, dτ = 2y dy, s goes from 0 to y^2
          Substitute s = u^2: ∫_0^y 2u * ... du where u ranges over sqrt(s)
          
        Integrand in y: ∫_0^y 2y' * exp(-r*(y'^2) - 0.5*d2^2) / (σ*sqrt(2π)) dy'
        where y' is integration variable over [0, y].
        """
        if y <= 0:
            return 0.0
        
        def integrand_y_prime(y_prime):
            """Integrand in y-space (integration variable y_prime)."""
            tau_prime = self._y_to_tau(y_prime)
            B_y_prime = rbf_interp(y_prime)
            
            if use_d1:
                d = self._d1(B_y, tau_prime, B_y_prime)
            else:
                d = self._d2(B_y, tau_prime, B_y_prime)
            
            # Factor 2*y_prime from ds = 2y dy conversion
            return 2.0 * y_prime * np.exp(-self.r * tau_prime - 0.5 * d * d) / (self.sigma * np.sqrt(2 * np.pi))
        
        return self._gk.integrate(integrand_y_prime, 0.0, y)
    
    def _integral_nonsingular_y_space(self, y, B_y, rbf_interp):
        """
        Compute non-singular integral in y-space.
        
        Original: ∫_0^τ exp(-r*s) * N(d1) ds
        
        In y-space: ∫_0^y 2*y' * exp(-r*y'^2) * N(d1(y'^2)) dy'
        """
        if y <= 0:
            return 0.0
        
        def integrand_y_prime(y_prime):
            """Integrand in y-space."""
            tau_prime = self._y_to_tau(y_prime)
            B_y_prime = rbf_interp(y_prime)
            d1 = self._d1(B_y, tau_prime, B_y_prime)
            
            # Factor 2*y_prime from ds = 2y dy conversion
            return 2.0 * y_prime * np.exp(-self.r * tau_prime) * norm.cdf(self._phi * d1)
        
        return self._gk.integrate(integrand_y_prime, 0.0, y)
    
    def _boundary_update_y_space(self, y, rbf_interp):
        """
        Update boundary at point y (in y-space) using RBF interpolator.
        
        Uses the RBF-interpolated boundary values for integration (previous estimate).
        
        Args:
            y: current y-space point
            rbf_interp: RBF interpolator for previous boundary
        
        Returns:
            updated boundary value
        """
        if y <= 0:
            return self._boundary_initial()
        
        tau = self._y_to_tau(y)
        B_y = rbf_interp(y)  # Get boundary at this point from RBF
        
        d1_tau = self._d1(B_y, tau, self.K)
        d2_tau = d1_tau - self.sigma * np.sqrt(tau)
        
        # Singular part (d2)
        term0 = (self.K / (self.sigma * np.sqrt(2 * np.pi * tau))) * np.exp(-self.r * tau - 0.5 * d2_tau * d2_tau)
        I2 = self._integral_singular_y_space(y, B_y, rbf_interp, use_d1=False)
        numerator = term0 + self.r * self.K * I2
        
        # Denominator
        denom0 = norm.cdf(self._phi * d1_tau) + (1.0 / (self.sigma * np.sqrt(2 * np.pi * tau))) * np.exp(-0.5 * d1_tau * d1_tau)
        
        if self.alpha <= 0:
            return numerator / denom0
        
        # With dividends
        denom_div = np.exp(-self.alpha * tau) * norm.cdf(self._phi * d1_tau)
        denom_div += (1.0 / (self.sigma * np.sqrt(2 * np.pi * tau))) * np.exp(-self.alpha * tau - 0.5 * d1_tau * d1_tau)
        
        I1 = self._integral_nonsingular_y_space(y, B_y, rbf_interp)
        I1_pdf = self._integral_singular_y_space(y, B_y, rbf_interp, use_d1=True)
        denom_div += self.alpha * (I1 + I1_pdf)
        
        return numerator / denom_div
    
    def compute_boundary_rbf(self, n_knots=20, n_iter=8, verbose=False):
        """
        Compute boundary using RBF in y-space with iteration.
        
        Args:
            n_knots: number of RBF knots in y-space
            n_iter: number of iterations
            verbose: print convergence info
        
        Returns:
            y_grid: y-space grid points
            B_grid: boundary values at grid points
            rbf_interp: final RBF interpolator
        """
        # Create knot grid in y-space (clustered near y=y_max)
        t_normalized = np.linspace(0, 1, n_knots)
        y_knots = self.y_max * t_normalized ** 1.5  # Cluster near y_max
        
        # Initial guess
        B_knots = np.full_like(y_knots, self._boundary_initial(), dtype=float)
        # Enforce boundary conditions:
        B_knots[-1] = self.K  # At y=y_max (τ=0), boundary is strike for put
        
        if n_iter <= 0:
            coeffs = self._build_rbf_system(y_knots, B_knots)
            rbf_interp = GaussianRBFInterpolator(y_knots, coeffs, self.shape)
            return y_knots, B_knots, rbf_interp
        
        # Iteration
        for iteration in range(n_iter):
            # Build RBF from current boundary values
            coeffs = self._build_rbf_system(y_knots, B_knots)
            rbf_interp = GaussianRBFInterpolator(y_knots, coeffs, self.shape)
            
            # Update boundary at knots
            B_new = np.zeros_like(B_knots)
            for i, y in enumerate(y_knots):
                B_raw = self._boundary_update_y_space(y, rbf_interp)
                # Damping: blend old and new (reduces oscillation)
                B_new[i] = 0.6 * B_knots[i] + 0.4 * B_raw
            
            # Sanitize: enforce constraints
            B_new = np.maximum(B_new, 0.01)  # Avoid exactly zero
            B_new = np.minimum(B_new, self.K)
            
            # For a put: boundary is non-increasing in y (since y -> 0 means τ -> ∞)
            # So enforce: B[i] >= B[i+1]
            for i in range(len(B_new) - 2, -1, -1):
                B_new[i] = max(B_new[i], B_new[i + 1])
            
            # Check convergence
            err = np.max(np.abs(B_new - B_knots))
            if verbose:
                print(f"  Iter {iteration}: max change = {err:.2e}, range = [{B_new.min():.6f}, {B_new.max():.6f}]")
            
            B_knots = B_new
            
            if err < 1e-8:
                if verbose:
                    print(f"  Converged after {iteration+1} iterations")
                break
        
        # Final RBF
        coeffs = self._build_rbf_system(y_knots, B_knots)
        rbf_interp = GaussianRBFInterpolator(y_knots, coeffs, self.shape)
        
        return y_knots, B_knots, rbf_interp
    
    def _continuation_put_value_y_space(self, S, y, rbf_interp):
        """
        Compute continuation value for put using RBF in y-space.
        
        Args:
            S: spot price
            y: time-to-maturity in y-space
            rbf_interp: RBF interpolator for boundary
        """
        tau = self._y_to_tau(y)
        euro = self._euro_put(S, tau)
        
        def integrand_y_prime(y_prime):
            """Integrand in y-space."""
            tau_prime = self._y_to_tau(y_prime)
            B_y_prime = rbf_interp(y_prime)
            d2 = self._d2(S, tau_prime, B_y_prime)
            
            term = self.r * self.K * np.exp(-self.r * tau_prime) * norm.cdf(-d2)
            if self.alpha > 0:
                d1 = self._d1(S, tau_prime, B_y_prime)
                term -= self.alpha * S * np.exp(-self.alpha * tau_prime) * norm.cdf(-d1)
            
            # Factor 2*y_prime from ds = 2y dy conversion
            return 2.0 * y_prime * term
        
        prem = self._gk.integrate(integrand_y_prime, 0.0, y)
        return euro + prem
    
    def price_put(self, S, tau, boundary_rbf=None):
        """
        Price American put.
        
        Args:
            S: spot price
            tau: time-to-maturity
            boundary_rbf: tuple (y_knots, B_knots, rbf_interp) or None (compute internally)
        """
        if tau <= 0:
            return max(self.K - S, 0.0)
        
        if boundary_rbf is None:
            y_knots, B_knots, rbf_interp = self.compute_boundary_rbf()
        else:
            y_knots, B_knots, rbf_interp = boundary_rbf
        
        y = self._tau_to_y(tau)
        B_tau = rbf_interp(y)
        
        if S <= B_tau:
            return self.K - S
        
        return self._continuation_put_value_y_space(S, y, rbf_interp)
    
    def price(self, S, tau, boundary_rbf=None):
        """Price American option (put only for now)."""
        if self.option_type == "put":
            return self.price_put(S, tau, boundary_rbf=boundary_rbf)
        else:
            raise NotImplementedError("Call pricing with RBF in y-space not yet implemented")


if __name__ == "__main__":
    # Test case
    pricer = RBFYSpaceAmerican(K=100, r=0.05, T=0.5, sigma=0.2, alpha=0.0, option_type="put")
    
    print("Computing boundary with RBF in y-space...")
    y_knots, B_knots, rbf_interp = pricer.compute_boundary_rbf(n_knots=15, n_iter=10, verbose=True)
    
    print(f"\ny-space knots (y_max={pricer.y_max:.6f}):")
    print(f"  y:   {y_knots}")
    print(f"  B(y): {B_knots}")
    
    print(f"\nBoundary RBF shape param: {pricer.shape:.6f}")
    print(f"RBF value at y=0:       {rbf_interp(0.0):.6f}")
    print(f"RBF value at y=y_max:   {rbf_interp(pricer.y_max):.6f}")
    
    print(f"\nPricing tests:")
    for S in [90, 100, 110]:
        price = pricer.price_put(S, 0.5, boundary_rbf=(y_knots, B_knots, rbf_interp))
        print(f"  Put(S={S}): {price:.8f}")
