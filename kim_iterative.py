import numpy as np
import math
from scipy.special import erf
from numba import njit
import time

# --- 1. CORE NUMBA KERNELS ---

@njit(fastmath=True)
def cubic_w(t):
    return np.array([-0.5*t**3 + t**2 - 0.5*t, 1.5*t**3 - 2.5*t**2 + 1.0, 
                     -1.5*t**3 + 2.0*t**2 + 0.5*t, 0.5*t**3 - 0.5*t**2])

@njit(fastmath=True)
def spline_lookup(z, y, z_grid, y_grid, vals, dz, dy):
    if z < z_grid[0]: return 0.0
    if z > z_grid[-1]: return 1.0
    zi_f = (z - z_grid[0]) / dz
    yi_f = (y - y_grid[0]) / dy
    iz = max(0, min(int(zi_f) - 1, len(z_grid) - 4))
    iy = max(0, min(int(yi_f) - 1, len(y_grid) - 4))
    tz, ty = zi_f - (iz + 1), yi_f - (iy + 1)
    wz, wy = cubic_w(tz), cubic_w(ty)
    res = 0.0
    for m in range(4):
        for k in range(4):
            res += vals[iz + m, iy + k] * wz[m] * wy[k]
    return res

@njit(fastmath=True)
def get_simpson_weights(tau):
    n = len(tau)
    w = np.zeros(n)
    if n < 2: return w
    if n == 2:
        dt = tau[1] - tau[0]
        return np.array([dt/2, dt/2])
    for i in range(0, n - 2, 2):
        h0, h1 = max(tau[i+1]-tau[i], 1e-15), max(tau[i+2]-tau[i+1], 1e-15)
        common = (h0 + h1) / (6 * h0 * h1)
        w[i] += common * h0 * (2 * h0 - h1)
        w[i+1] += common * (h0 + h1)**2
        w[i+2] += common * h1 * (2 * h1 - h0)
    if n % 2 == 0:
        dt = tau[-1] - tau[-2]
        w[-1] += dt/2; w[-2] += dt/2
    return w

@njit(fastmath=True)
def get_trapezoid_weights(tau):
    n = len(tau)
    w = np.zeros(n)
    if n < 2: return w
    
    # First point
    w[0] = 0.5 * (tau[1] - tau[0])
    
    # Internal points
    for i in range(1, n - 1):
        w[i] = 0.5 * (tau[i+1] - tau[i-1])
        
    # Last point
    w[n-1] = 0.5 * (tau[n-1] - tau[n-2])
    
    return w

@njit(fastmath=True)
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))

@njit(fastmath=True)
def eep_fused_kernel(S, B_hist, tau_vec, eq, er, dt_w, z_g, y_g, vals, dz, dy, sigma, r, q, K, w_flag, use_spline):
    total = 0.0
    drift = (r - q - 0.5 * sigma**2)
    for i in range(len(B_hist)):
        vst = sigma * np.sqrt(max(tau_vec[i], 1e-12))
        z_d2 = (np.log(S / B_hist[i]) + drift * tau_vec[i]) / vst
        if use_spline:
            nd1 = spline_lookup(w_flag * (z_d2 + vst), vst, z_g, y_g, vals, dz, dy)
            nd2 = spline_lookup(w_flag * z_d2, vst, z_g, y_g, vals, dz, dy)
        else:
            nd1 = norm_cdf(w_flag * (z_d2 + vst))
            nd2 = norm_cdf(w_flag * z_d2)
        total += (w_flag * q * S * eq[i] * nd1 - w_flag * r * K * er[i] * nd2) * dt_w[i]
    return total

# --- 2. SOLVER CLASS ---

class KimSplineV14:
    def __init__(self, S0, K, sigma, T, r, q, w=1, divs=None):
        self.S0, self.K, self.sigma, self.T, self.r, self.q, self.w = S0, K, sigma, T, r, q, w
        self.divs = divs if divs else {}
        z_grid = np.linspace(-40, 40, 500)
        y_grid = np.linspace(0, sigma * np.sqrt(T) + 0.1, 120)
        self.dz, self.dy, self.z_g, self.y_g = z_grid[1]-z_grid[0], y_grid[1]-y_grid[0], z_grid, y_grid
        ZZ, _ = np.meshgrid(z_grid, y_grid, indexing='ij')
        self.vals = 0.5 * (1 + erf(ZZ / np.sqrt(2)))

    def solve(self, knots_per_seg=11, use_spline=True, integration_method='simpson'):
        start_time = time.time()
        div_times = sorted([t for t in self.divs.keys() if 0 < t < self.T], reverse=True)
        endpoints = sorted([0, self.T] + div_times, reverse=True)
        all_t, all_B, all_w = [], [], []
        
        do_speedy = True

        for i in range(len(endpoints)-1):
            t_hi, t_lo = endpoints[i], endpoints[i+1]
            t_seg = t_hi - np.linspace(0, np.sqrt(t_hi - t_lo), knots_per_seg)**2
            B_seg = np.zeros(knots_per_seg)
            
            # Anchor junction to previous segment result
            B_seg[0] = (all_B[-1] if i > 0 else self.K)
            if i > 0: B_seg[0] += self.w * self.divs.get(t_hi, 0.0)

            for j in range(1, knots_per_seg):
                curr_t = t_seg[j]
                tau_mat = max(self.T - curr_t, 1e-15)
                pv_divs = sum(amt * np.exp(-self.r*(dt-curr_t)) for dt, amt in self.divs.items() if dt > curr_t)
                
                # History Setup
                t_curr_sub = t_seg[:j+1]
                
                if integration_method == 'trapezoid':
                     w_curr_sub = get_trapezoid_weights(np.sort(t_curr_sub - curr_t))
                else: 
                     w_curr_sub = get_simpson_weights(np.sort(t_curr_sub - curr_t))
                
                h_t = np.concatenate((np.array(all_t), t_curr_sub[:-1]))
                h_B = np.concatenate((np.array(all_B), B_seg[:j]))
                
                # FIX: weights need to be reversed to match descending history
                # all_w stores weights for segments (already reversed when stored)
                # w_curr_sub is ascending (0 to tau), we need [:-1] reversed for history
                dt_w = np.concatenate((np.array(all_w), w_curr_sub[:-1][::-1]))
                
                tau_v = h_t - curr_t
                vst = self.sigma * np.sqrt(tau_mat)
                eq_mat, er_mat = np.exp(-self.q*tau_mat), np.exp(-self.r*tau_mat)

                # --- STABILIZED NEWTON (Choice 2 & 3) ---
                u = np.log(max(B_seg[j-1], 1e-4))
                for _ in range(50):
                    # Hard corridor for the first knot of a segment
                    if j == 1:
                        u = np.clip(u, np.log(B_seg[0]*0.95), np.log(B_seg[0]*1.05))
                    else:
                        u = np.clip(u, 2.0, 7.0)

                    x = np.exp(u)
                    S_tilde = max(x - self.w * pv_divs, 1e-4)
                    d1 = (np.log(S_tilde/self.K) + (self.r-self.q+0.5*self.sigma**2)*tau_mat)/max(vst, 1e-12)
                    if use_spline:
                        nd1 = spline_lookup(self.w*d1, vst, self.z_g, self.y_g, self.vals, self.dz, self.dy)
                        nd2 = spline_lookup(self.w*(d1-vst), vst, self.z_g, self.y_g, self.vals, self.dz, self.dy)
                    else:
                        nd1 = norm_cdf(self.w*d1)
                        nd2 = norm_cdf(self.w*(d1-vst))
                    
                    euro = self.w * (S_tilde * eq_mat * nd1 - self.K * er_mat * nd2)
                    eep = eep_fused_kernel(S_tilde, h_B, tau_v, np.exp(-self.q*tau_v), 
                                           np.exp(-self.r*tau_v), dt_w, self.z_g, self.y_g, 
                                           self.vals, self.dz, self.dy, self.sigma, self.r, self.q, self.K, self.w, use_spline)
                    
                    f = (self.w * (x - self.K)) - (euro + eep)
                    if abs(f) < 1e-9: break
                    
                    df = (self.w - self.w * eq_mat * nd1) * x
                    if abs(df) < 1e-7: break # Gradient vanished, stop drifting

                    # LEASH: Dynamic Damping
                    alpha = 0.05 if (i == 0 and j < 5) else 0.35
                    if do_speedy:
                        alpha = 0.95
                    u = u - alpha * f / (max(df, 1e-8))
                
                B_seg[j] = np.exp(u)
            
            # Freeze segment
            if integration_method == 'trapezoid':
                 seg_w_final = get_trapezoid_weights(np.sort(t_seg - t_lo))
            else:
                 seg_w_final = get_simpson_weights(np.sort(t_seg - t_lo))
            
            # FIX: Reverse weights to match descending time order in all_t
            all_t.extend(t_seg[:-1]); all_B.extend(B_seg[:-1]); all_w.extend(seg_w_final[:-1][::-1])

        return self._finalize(np.array(all_t), np.array(all_B), start_time, use_spline, integration_method)

    def _finalize(self, t_f, B_f, start_t, use_spline, integration_method):
        idx = np.argsort(t_f); ts, Bs = t_f[idx], B_f[idx]; m = ts > 1e-9
        pv_divs0 = sum(amt * np.exp(-self.r * dt) for dt, amt in self.divs.items())
        S_tilde0 = max(self.S0 - self.w * pv_divs0, 1e-4)
        vst0 = self.sigma * np.sqrt(self.T)
        d1_0 = (np.log(S_tilde0/self.K) + (self.r-self.q+0.5*self.sigma**2)*self.T) / vst0
        if use_spline:
            nd1_0 = spline_lookup(self.w * d1_0, vst0, self.z_g, self.y_g, self.vals, self.dz, self.dy)
            nd2_0 = spline_lookup(self.w * (d1_0-vst0), vst0, self.z_g, self.y_g, self.vals, self.dz, self.dy)
        else:
            nd1_0 = norm_cdf(self.w * d1_0)
            nd2_0 = norm_cdf(self.w * (d1_0-vst0))
        euro0 = self.w * (S_tilde0 * np.exp(-self.q*self.T) * nd1_0 - self.K * np.exp(-self.r*self.T) * nd2_0)
        
        if integration_method == 'trapezoid':
             dt_w0 = get_trapezoid_weights(ts[m])
        else:
             dt_w0 = get_simpson_weights(ts[m])

        eep0 = eep_fused_kernel(S_tilde0, Bs[m], ts[m], np.exp(-self.q*ts[m]), 
                                np.exp(-self.r*ts[m]), dt_w0, self.z_g, self.y_g, 
                                self.vals, self.dz, self.dy, self.sigma, self.r, self.q, self.K, self.w, use_spline)
        return {'time': t_f, 'euro': euro0, 'eep': eep0, 'amer': euro0 + eep0, 'boundary': B_f, 'runtime': time.time() - start_t}

# --- 3. RUN ---

if __name__ == '__main__':
    import matplotlib
    try:
        matplotlib.use('Agg')
    except:
        pass
    import matplotlib.pyplot as plt

    divs = {}#{0.1: 0.0, 0.2: 0.0}
    params = {'S0': 100, 'K': 100, 'sigma': 0.10, 'T': 0.25, 'r': 0.03, 'q': 0.03, 'w': -1, 'divs': divs}
    res = KimSplineV14(**params).solve(knots_per_seg=21, use_spline=True, integration_method='trapezoid')
    # res = KimSplineV14(**params).solve(knots_per_seg=11, use_spline=True)
    
    print(f"Amer: {res['amer']:.5f} (Euro: {res['euro']:.5f}, EEP: {res['eep']:.5f})")
    
    plt.figure(figsize=(10, 4))
    plt.plot(res['time'], res['boundary'], 'o-', markersize=2, label="V14 Stabilized")
    plt.axhline(100, color='k', alpha=0.2, linestyle='--')
    plt.title("V14: Continuity Anchored + Leashed Newton")
    # plt.ylim(90,150)
    plt.legend(); plt.grid(alpha=0.2)
    plt.savefig('boundary_output.png')
    print("Plot saved to boundary_output.png")