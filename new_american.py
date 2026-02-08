import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy.optimize import brentq
from scipy.special import erf
from scipy.stats import norm


# ============================================================================
# RBF Kernel Classes
# ============================================================================

class RBFKernel:
    """Base class for radial basis function kernels."""
    
    def __init__(self, shape_parameter):
        """Initialize with shape parameter c."""
        self.c = shape_parameter
    
    def phi(self, r):
        """Evaluate the kernel φ(r)."""
        raise NotImplementedError
    
    def G(self, z):
        """Evaluate the integral transform G(z) = ∫₀ᶻ φ(t) dt."""
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}(c={self.c:.4f})"


class MultiQuadric(RBFKernel):
    """Multi-quadric kernel: φ(r) = √(r² + c²)"""
    
    def phi(self, r):
        return np.sqrt(r**2 + self.c**2)
    
    def G(self, z):
        """∫₀ᶻ √(t² + c²) dt = (z√(z²+c²) + c²·sinh⁻¹(z/c)) / 2"""
        return (z * np.sqrt(z**2 + self.c**2) + self.c**2 * np.arcsinh(z / self.c)) / 2


class InverseMultiQuadric(RBFKernel):
    """Inverse multi-quadric kernel: φ(r) = 1/√(r² + c²)"""
    
    def phi(self, r):
        return 1.0 / np.sqrt(r**2 + self.c**2)
    
    def G(self, z):
        """∫₀ᶻ 1/√(t² + c²) dt = sinh⁻¹(z/c)"""
        return np.arcsinh(z / self.c)


class Gaussian(RBFKernel):
    """Gaussian kernel: φ(r) = exp(-(c·r)²)"""
    
    def phi(self, r):
        return np.exp(-(self.c * r)**2)
    
    def G(self, z):
        """∫₀ᶻ exp(-c²·t²) dt = √π/(2c)·erf(c·z)"""
        return np.sqrt(np.pi) / (2 * self.c) * erf(self.c * z)


# ============================================================================
# Warm Start Initialization
# ============================================================================

class JuWarmStart:
    """Minimized helper to generate initial boundary guess using Ju (1998)."""

    @staticmethod
    def get_initial_H(pricer):
        S_star_T = JuWarmStart._solve_ju_critical_price(pricer)

        y_max = np.sqrt(pricer.T)
        a0 = np.log(pricer.B0)
        a1 = (np.log(S_star_T) - a0) / y_max

        initial_H = a0 + a1 * pricer.y[1:]
        return initial_H

    @staticmethod
    def _solve_ju_critical_price(p):
        phi = 1 if p.w == 1 else -1
        h = 1 - np.exp(-p.r * p.T)
        alpha = 2 * p.r / p.sigma**2
        beta = 2 * (p.r - p.q) / p.sigma**2

        term = (beta - 1) ** 2 + 4 * alpha / max(h, 1e-7)
        lam = (-(beta - 1) + phi * np.sqrt(term)) / 2

        def objective(S_star):
            d1 = (np.log(S_star / p.K) + (p.r - p.q + 0.5 * p.sigma**2) * p.T) / (p.sigma * np.sqrt(p.T))
            d2 = d1 - p.sigma * np.sqrt(p.T)
            ve = phi * (
                S_star * np.exp(-p.q * p.T) * norm.cdf(phi * d1)
                - p.K * np.exp(-p.r * p.T) * norm.cdf(phi * d2)
            )

            prob_term = phi * np.exp(-p.q * p.T) * norm.cdf(phi * d1)
            rhs = prob_term + lam * (phi * (S_star - p.K) - ve) / S_star
            return phi - rhs

        if p.w == 1:
            low, high = p.K, p.K * 10
        else:
            low, high = 1e-4, p.K

        try:
            return brentq(objective, low, high)
        except Exception:
            return p.B0 * 1.2 if p.w == 1 else p.B0 * 0.8


class AmericanRBFEngine:
    def __init__(self, K, T, r, q, sigma, w=1, n_knots=16, alpha=1e-11, rbf_kernel=None):
        self.K, self.T, self.r, self.q, self.sigma, self.w = K, T, r, q, sigma, w
        self.alpha = alpha
        self.y = np.linspace(0, np.sqrt(T), n_knots)
        self.N = n_knots
        self.B0 = K
        
        # Set default kernel if not provided
        if rbf_kernel is None:
            dy = np.mean(np.diff(self.y))
            rbf_kernel = InverseMultiQuadric(shape_parameter=2.0 * dy)
        self.rbf = rbf_kernel
        
        self.A = self._precompute_operator()

    def _precompute_operator(self):
        coords = self.y[:, np.newaxis]
        R = np.abs(coords - coords.T)

        Phi = self.rbf.phi(R)
        upper, lower = self.y[:, np.newaxis] - self.y[np.newaxis, :], -self.y[np.newaxis, :]
        Psi = self.rbf.G(upper) - self.rbf.G(lower)

        return Psi @ inv(Phi.T @ Phi + self.alpha * np.eye(self.N)) @ Phi.T

    def _black_scholes(self, S, K, tau):
        tau = np.maximum(tau, 1e-12)
        v_sqrt_t = self.sigma * np.sqrt(tau)
        d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / v_sqrt_t
        d2 = d1 - v_sqrt_t
        p = self.w * (
            S * np.exp(-self.q * tau) * norm.cdf(self.w * d1)
            - K * np.exp(-self.r * tau) * norm.cdf(self.w * d2)
        )
        d = self.w * np.exp(-self.q * tau) * norm.cdf(self.w * d1)
        return p, d

    def _eep_integrand(self, S_spot, B_strike, tau, u):
        v_sqrt_tau = self.sigma * np.sqrt(tau)
        d1 = (np.log(S_spot / B_strike) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / v_sqrt_tau
        d2 = d1 - v_sqrt_tau
        exp_q, exp_r = np.exp(-self.q * tau), np.exp(-self.r * tau)

        f = 2 * u * self.w * (
            self.q * S_spot * exp_q * norm.cdf(self.w * d1)
            - self.r * self.K * exp_r * norm.cdf(self.w * d2)
        )
        return f, d1, exp_q

    def get_residual_and_jac(self, H):
        B = np.concatenate([[self.B0], np.exp(H)])
        N = len(B)

        B_spot = B[:, np.newaxis]
        B_strike = B[np.newaxis, :]
        tau_mat = np.maximum(self.y[:, np.newaxis]**2 - self.y[np.newaxis, :]**2, 1e-10)
        mask = (self.y[:, np.newaxis]**2 - self.y[np.newaxis, :]**2) > 1e-12

        f_mat, d1, exp_q = self._eep_integrand(B_spot, B_strike, tau_mat, self.y[np.newaxis, :])
        euro_p, euro_d = self._black_scholes(B, self.K, self.y**2)

        eep_full_matrix = self.A @ f_mat.T
        eep = np.diag(eep_full_matrix)
        res = (self.w * (B - self.K) - (euro_p + eep))[1:]

        t1 = (-2 * self.y[np.newaxis, :] * B_spot * exp_q * norm.pdf(d1)) / (
            B_strike * self.sigma * np.sqrt(tau_mat)
        )
        t2 = self.q - (self.r * self.K / B_strike) * np.exp(-(self.r - self.q) * tau_mat)
        df_dB_strike = np.where(mask, t1 * t2, 0)

        exp_r = np.exp(-self.r * tau_mat)
        dd1_dS = 1.0 / (B_spot * self.sigma * np.sqrt(tau_mat))

        term1 = 2 * self.y[np.newaxis, :] * self.w * self.q * exp_q * norm.cdf(self.w * d1)
        term2 = 2 * self.y[np.newaxis, :] * self.w * self.w * dd1_dS * (
            self.q * B_spot * exp_q * norm.pdf(d1)
            - self.r * self.K * exp_r * norm.pdf(d1 - self.sigma * np.sqrt(tau_mat))
        )
        df_dB_spot = np.where(mask, term1 + term2, 0)

        J_lin = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                term_strike = -self.A[i, j] * df_dB_strike[i, j]

                term_spot = 0
                if i == j:
                    term_spot = -np.sum(self.A[i, :] * df_dB_spot[i, :])

                term_exercise = 0
                if i == j:
                    term_exercise = self.w - euro_d[i]

                J_lin[i, j] = term_strike + term_spot + term_exercise

        J_log = J_lin[1:, 1:] * B[1:][np.newaxis, :]
        return res, J_log

    def solve_boundary(self, max_iters):
        H = JuWarmStart.get_initial_H(self)

        history = {}
        for i in range(max_iters):
            res, jac = self.get_residual_and_jac(H)

            history[f"Iter{i:2d}"] = np.concatenate([[self.B0], np.exp(H)])
            if np.linalg.norm(res) < 1e-9:
                break
            H -= np.linalg.solve(jac, res)
            H = np.clip(a=H, a_min=np.log(self.K / 2.0), a_max=np.log(self.K * 2.0))
        return np.concatenate([[self.B0], np.exp(H)]), history

    def price(self, S0, max_iters=7):
        start_time = time.perf_counter()
        B_final, history = self.solve_boundary(max_iters=max_iters)

        euro_price, _ = self._black_scholes(S0, self.K, self.T)

        tau_eep = np.maximum(1e-12, self.T - self.y**2)
        f_vec, _, _ = self._eep_integrand(S0, B_final, tau_eep, self.y)
        eep_val = max(0, self.A[-1, :] @ f_vec)

        runtime = time.perf_counter() - start_time

        return {
            "euro": float(euro_price),
            "eep": float(eep_val),
            "amer": float(euro_price + eep_val),
            "runtime": runtime,
            "history": history,
            "boundary": pd.DataFrame({"B": B_final}, index=self.y**2),
        }


def test_me(S0, K, T, r, q, sigma, w, n_knots, max_iters, rbf_kernel=None):
    pricer = AmericanRBFEngine(K=K, T=T, r=r, q=q, sigma=sigma, w=w, n_knots=n_knots, rbf_kernel=rbf_kernel)
    results = pricer.price(S0=S0, max_iters=max_iters)

    ls = ["euro", "eep", "amer", "runtime"]
    str_ = ""
    for k in ls:
        str_ += f"{k}: {results[k]:.8f} "
    print(f"Kernel: {pricer.rbf} | {str_}")

    # Plot 1: Boundary Convergence
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    for i, b in results["history"].items():
        plt.plot(pricer.y**2, b, label=f"Iter {i}", alpha=0.5)
    last_boundary = list(results["history"].values())[-1]
    plt.plot(pricer.y**2, last_boundary, "kx", markersize=3, alpha=0.7, label="_nolegend_")
    plt.title("Fast Boundary Convergence")
    plt.xlabel("τ = t²")
    plt.ylabel("Boundary B(τ)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.3)
    
    # Plot 2: Individual RBFs
    plt.subplot(1, 2, 2)
    y_fine = np.linspace(0, np.sqrt(T), 200)
    
    for i, y_center in enumerate(pricer.y):
        r = np.abs(y_fine - y_center)
        rbf_vals = pricer.rbf.phi(r)
        plt.plot(y_fine**2, rbf_vals, label=f"RBF {i}", alpha=0.6)
    
    plt.plot(pricer.y**2, np.ones_like(pricer.y), "kx", markersize=8, label="Knots")
    plt.title(f"Individual {pricer.rbf.__class__.__name__} Basis Functions")
    plt.xlabel("τ = t²")
    plt.ylabel("φ(r)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("boundary_convergence.png")
    print("Saved boundary_convergence.png")


if __name__ == "__main__":
    S0, K, T, r, q, sigma, w, n_knots, max_iters = 100, 100, 1.0, 0.05, 0.05, 0.25, +1, 5, 6
    
    # Test with default Inverse Multi-Quadric
    print("\n=== Inverse Multi-Quadric (default) ===")
    test_me(S0, K, T, r, q, sigma, w, n_knots, max_iters)
    
    # Test with Multi-Quadric
    print("\n=== Multi-Quadric ===")
    dy = np.mean(np.diff(np.linspace(0, np.sqrt(T), n_knots)))
    mq = MultiQuadric(shape_parameter=2.0 * dy)
    test_me(S0, K, T, r, q, sigma, w, n_knots, max_iters, rbf_kernel=mq)
    
    # Test with Gaussian
    print("\n=== Gaussian ===")
    gauss = Gaussian(shape_parameter=1.0 / dy)
    test_me(S0, K, T, r, q, sigma, w, n_knots, max_iters, rbf_kernel=gauss)
