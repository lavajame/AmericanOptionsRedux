
import numpy as np
from scipy.stats import norm
from scipy.optimize import root

class Ju1998Piecewise:
    def __init__(self, S, K, r, T, sigma, delta, option_type='put'):
        self.option_type = option_type.lower()
        if self.option_type == 'call':
            # Use symmetry: Call(S, K, r, d, sg, T) = Put(K, S, d, r, sg, T)
            self.S = K
            self.K = S
            self.r = delta
            self.delta = r
        else:
            self.S = S
            self.K = K
            self.r = r
            self.delta = delta
            
        self.T = T
        self.sigma = sigma
        # For Put: intrinsic = K - S, delta = -1
        # Everything below is for American Put

    def _d1(self, S_val, T_val):
        if T_val <= 1e-9: return np.inf if S_val > 0 else -np.inf
        return (np.log(S_val / self.K) + (self.r - self.delta + 0.5 * self.sigma**2) * T_val) / (self.sigma * np.sqrt(T_val))

    def _d2(self, S_val, T_val):
        return self._d1(S_val, T_val) - self.sigma * np.sqrt(T_val)

    def european_put_price(self, S_val, T_val):
        if T_val <= 1e-9: return max(0, self.K - S_val)
        d1 = self._d1(S_val, T_val)
        d2 = d1 - self.sigma * np.sqrt(T_val)
        return self.K * np.exp(-self.r * T_val) * norm.cdf(-d2) - S_val * np.exp(-self.delta * T_val) * norm.cdf(-d1)

    def european_put_delta(self, S_val, T_val):
        if T_val <= 1e-9: return -1.0 if S_val < self.K else 0.0
        d1 = self._d1(S_val, T_val)
        return np.exp(-self.delta * T_val) * (norm.cdf(d1) - 1)

    def _I_func(self, t1, t2, S_inp, B_inp, b_inp, xi_inp):
        # Calculates I(t1, t2, S, B, b, -1, xi)
        
        mu_prime = (self.r - self.delta - b_inp + (0.5 * self.sigma**2 if xi_inp == self.delta else -0.5 * self.sigma**2))
        z1 = mu_prime / self.sigma
        z2 = np.log(S_inp / B_inp) / self.sigma
        z3 = np.sqrt(z1**2 + 2 * xi_inp)
        if z3 < 1e-9: z3 = 1e-9
        
        sqrt_t1 = np.sqrt(t1)
        sqrt_t2 = np.sqrt(t2)
        
        def N(x): return norm.cdf(x)
        
        # Standard terms
        def n_term(z3, z2, t):
            if t < 1e-9: return 1.0 if z2 > 0 else 0.0 
            return N(z3 * np.sqrt(t) + z2 / np.sqrt(t))
            
        def n_term_minus(z3, z2, t):
             if t < 1e-9: return 1.0 if -z2 > 0 else 0.0
             return N(z3 * np.sqrt(t) - z2 / np.sqrt(t))

        term1 = np.exp(xi_inp * t1) * n_term(z1, z2, t1)
        term2 = np.exp(xi_inp * t2) * n_term(z1, z2, t2)
        
        # Complex terms with potential overflow
        def safe_prod(E, x, y):
            # Returns exp(E) * (N(x) - N(y))
            if E < 100:
                return np.exp(E) * (norm.cdf(x) - norm.cdf(y))
                
            # Asymptotic expansion for large E
            # Check sign of x, y.
            if x > 5 and y > 5:
                # Use tails: N(-y) - N(-x)
                log_n_y = -0.5 * y**2 - np.log(y * np.sqrt(2*np.pi))
                log_n_x = -0.5 * x**2 - np.log(x * np.sqrt(2*np.pi))
                t_y = np.exp(E + log_n_y) if (E + log_n_y) < 700 else 0 
                t_x = np.exp(E + log_n_x) if (E + log_n_x) < 700 else 0
                return t_y - t_x
            
            if x < -5 and y < -5:
                # Direct Mills
                log_n_y = -0.5 * y**2 - np.log(abs(y) * np.sqrt(2*np.pi))
                log_n_x = -0.5 * x**2 - np.log(abs(x) * np.sqrt(2*np.pi))
                t_y = np.exp(E + log_n_y) 
                t_x = np.exp(E + log_n_x)
                return t_x - t_y
            
            # If mixed or small, return infinity if E is huge (failure case for formula)
            return np.inf 

        def eval_term(scale_exp, z_coeff, sign_z2):
            arg1 = z3 * sqrt_t1 + sign_z2 * z2 / sqrt_t1
            arg2 = z3 * sqrt_t2 + sign_z2 * z2 / sqrt_t2
            return 0.5 * z_coeff * safe_prod(scale_exp, arg2, arg1)

        # Pair 1: (z1/z3 + 1), exp(z2(z3-z1)), N(+ arg)
        exp_1 = z2 * (z3 - z1)
        term_A = eval_term(exp_1, (z1/z3 + 1), 1.0)
        
        # Pair 2: (z1/z3 - 1), exp(z2(z3+z1)), N(- arg)
        exp_2 = z2 * (z3 + z1)
        term_B = eval_term(exp_2, (z1/z3 - 1), -1.0)
        
        return (np.exp(-xi_inp * t1) * n_term(z1, z2, t1) - 
                np.exp(-xi_inp * t2) * n_term(z1, z2, t2) +
                term_A + term_B)

    def _I_S_func(self, t1, t2, S_inp, B_inp, b_inp, xi_inp):
        # Derivative w.r.t S. Eq 1618.
        # Terms involving n(pdf).
        pass # To implement if needed for delta.
        # Actually explicit formula for Delta might be better?
        # Or Finite Difference delta inside solver?
        # Solver needs residuals.
        # Residual 2: Delta - (-1) = 0.
        # Put Delta = European Delta + Premium Delta.
        # Premium = rK I1 - delta S I2.
        # dPremium/dS = rK dI1/dS - delta (I2 + S dI2/dS).
        # Need dI/dS.
        # dI/dS uses Eq 18 (Line 1611).
        
        mu_prime = (self.r - self.delta - b_inp + (0.5 * self.sigma**2 if xi_inp == self.delta else -0.5 * self.sigma**2))
        z1 = mu_prime / self.sigma
        z2 = np.log(S_inp / B_inp) / self.sigma
        z3 = np.sqrt(z1**2 + 2 * xi_inp)
        
        def n_pdf(x): return norm.pdf(x)
        
        def little_n(z_a, z_b, t):
            if t < 1e-9: return 0.0
            arg = z_a * np.sqrt(t) + z_b / np.sqrt(t)
            return n_pdf(arg)

        def little_n_minus(z_a, z_b, t):
             if t < 1e-9: return 0.0
             arg = z_a * np.sqrt(t) - z_b / np.sqrt(t)
             return n_pdf(arg)
             
        def big_N(z_a, z_b, t):
             if t < 1e-9: return 1.0 if z_b > 0 else 0.0
             arg = z_a * np.sqrt(t) + z_b / np.sqrt(t)
             return N(arg)

        def big_N_minus(z_a, z_b, t):
             if t < 1e-9: return 1.0 if -z_b > 0 else 0.0
             arg = z_a * np.sqrt(t) - z_b / np.sqrt(t)
             return N(arg)

        # Formula Eq 18 is complex.
        # Common factor 1/(S sigma).
        # Term 1: e^... n(...)
        # I'll implement finite difference for S derivative to avoid complexity and errors.
        eps = S_inp * 1e-5
        val_plus = self._I_func(t1, t2, S_inp + eps, B_inp, b_inp, xi_inp)
        val_minus = self._I_func(t1, t2, S_inp - eps, B_inp, b_inp, xi_inp)
        return (val_plus - val_minus) / (2 * eps)

    def _solve_segment(self, t_match, T_calc, known_segments, prev_B=None, prev_b=None):
        # Solves for B_curr, b_curr for interval [0, t_match] (relative to T_calc)
        # Actually T_calc is the "Maturity" for this calculation.
        # t_match is the point where we enforce conditions (e.g. 0 or T/2).
        # Interval of unknown boundary is [0, t_match]? NO.
        # Recursion:
        # P1: Maturity T. Match at 0. Unknown [0, T]. Known: None.
        # P2: Step 1: Maturity T/2. Match at 0 (relative). Unknown [0, T/2]. Known: None. -> B21, b21.
        #     Step 2: Maturity T. Match at 0. Unknown [0, T/2]. Known: [T/2, T] (shifted B21). -> B22, b22.
        
        # Correct Logic:
        # Segment 1 (Latest): B_21. Valid on [T/2, T].
        # We find it by conceptualizing option with maturity T/2? 
        # Eq 14: K - B = P_E(T/2) + ...
        # This implies we solve for B21 by treating it as B11 for T/2?
        # "To find B21... B=B11... are good initial values".
        # Yes, calculate parameters for segment of length L as if it were a 1-piece match.
        
        # General Solver:
        # Solve for B, b such that at time `t_match` (usually 0),
        # Price(S=B) = K - B
        # Delta(S=B) = -1
        # The Price function integrates from 0 to T_calc.
        # Segments:
        # [0, L] -> Unknown (B, b).
        # [L, L+Li] -> Known.
        
        # Integration split:
        # Int[0, T_calc] = Int[0, L] (Unknown) + Sum Int[known].
        
        # L = len(Unknown) = T_calc - sum(len(known)).
        # t_match is usually 0.
        
        def objectives(x):
            # x[0] = log(B), x[1] = b
            log_B, b_try = x
            B_try = np.exp(log_B)
            
            if B_try > self.K * 2.0: return [1e9, 1e9] # Loose upper bound
            
            # Compute Premium
            # 1. Unknown segment [0, L]
            # Since we match at 0, interval is [0, L].
            # L is determined by what?
            # If P1, L=T.
            # If P2 step 2, L=T/2. Known is [T/2, T].
            
            L = T_calc
            if known_segments:
                L = known_segments[0]['start'] # Assuming ordered?
                # Actually, pass L explicitly or deduce.
            
            # Premium from current segment [0, L]
            # I1(0, L, ...), I2(0, L, ...)
            
            # rK * I1 - delta * S * I2
            # For I1: xi=r. I2: xi=delta.
            
            # Term for current
            # Note: S=B_try.
            # And inputs to I need S, B, b.
            # B, b are B_try, b_try.
            
            val_I1 = self._I_func(0, L, B_try, B_try, b_try, self.r)
            val_I2 = self._I_func(0, L, B_try, B_try, b_try, self.delta)
            
            prem = self.r * self.K * val_I1 - self.delta * B_try * val_I2
            
            # Terms for known segments
            for seg in known_segments:
                # seg has 'B', 'b', 'start', 'end'
                # Integration limits [start, end]
                # B(t) = B_seg * exp(b_seg * (t - start)) ??
                # Paper: B_t = B_{21} e^{b_{21} t} ...
                # Wait. "t" in integral is time from "now" (0).
                # Known segment B is defined relative to ITS start?
                # Paper: B_{21} e^{b_{21} t}.
                # The exponent applies to time t from 0.
                # So B and b are absolute.
                
                s_s, s_e = seg['start'], seg['end']
                s_B, s_b = seg['B'], seg['b']
                
                v_I1 = self._I_func(s_s, s_e, B_try, s_B, s_b, self.r)
                v_I2 = self._I_func(s_s, s_e, B_try, s_B, s_b, self.delta)
                prem += self.r * self.K * v_I1 - self.delta * B_try * v_I2
            
            eu_price = self.european_put_price(B_try, T_calc)
            price = eu_price + prem
            
            # Delta
            # diff price / diff S at S=B_try
            # Finite diff
            eps = 1e-4 * B_try
            # Recalculate price at S = B_try + eps
            
            # ... refactor price calc to func
            def calc_p(St):
                p_i1 = self._I_func(0, L, St, B_try, b_try, self.r)
                p_i2 = self._I_func(0, L, St, B_try, b_try, self.delta)
                pr = self.r * self.K * p_i1 - self.delta * St * p_i2
                
                for seg in known_segments:
                    s_s, s_e = seg['start'], seg['end']
                    s_B, s_b = seg['B'], seg['b']
                    v1 = self._I_func(s_s, s_e, St, s_B, s_b, self.r)
                    v2 = self._I_func(s_s, s_e, St, s_B, s_b, self.delta)
                    pr += self.r * self.K * v1 - self.delta * St * v2
                    
                return self.european_put_price(St, T_calc) + pr

            p_plus = calc_p(B_try + eps)
            p_minus = calc_p(B_try - eps)
            delta_val = (p_plus - p_minus) / (2 * eps)
            
            return [price - (self.K - B_try), delta_val + 1.0]

        # Initial guess
        if prev_B is None:
            # Simple heuristic
            x0 = [np.log(self.K * 0.9), 0.0] 
        else:
            x0 = [np.log(prev_B), prev_b]
            
        sol = root(objectives, x0, method='hybr')
        return np.exp(sol.x[0]), sol.x[1]

    def compute_price_N(self, N):
        # 1. Determine segments
        # Time points
        dt = self.T / N
        segments = [] # List of dicts
        
        # Base B, b guesses
        last_B = self.K
        last_b = 0.0
        
        # Loop backwards?
        # Paper implies recursive construction.
        # Find B_{NN}, b_{NN} (Last segment [T-dt, T] - relative to 0? No.)
        # Paper Notation: B_{21} is later segment [T/2, T].
        # We calculate B_{21} first.
        # Calculation for B_{21} corresponds to P1 problem with T'=T/2? No, T'=T-(T/2) = T/2.
        # The segment is "at the end".
        # It's an approximation of the boundary near expiry.
        # So we solve a 1-period problem with maturity dt.
        # Then 2-period problem? No.
        # We solve 1-period problem with maturity dt. -> This gives properties of segment near expiry.
        # We map it to [T-dt, T].
        # Then solve for next segment [T-2dt, T-dt], knowing the last one.
        
        # Correct Loop:
        # For k = 1 to N:
        #   Calc segment valid for duration k*dt.
        #   Segment interval in integral is [0, dt].
        #   But we append it to "future".
        #   Known segments are shifted.
        
        # Wait, integral uses absolute time t from 0 to T.
        # If we solve for segment N (closest to expiry, time T-dt to T):
        # We treat "now" as T-dt. Maturity is dt.
        # Solve P1(dt). Get B, b.
        # This segment is B(t') = B e^{b t'} for t' in [0, dt].
        # In absolute time t (from 0), this is valid for t in [T-dt, T].
        # t' = t - (T-dt).
        # B(t) = B * exp(b * (t - (T-dt))).
        # We can convert this to B_abs * exp(b * t).
        # B_abs = B * exp(-b * (T-dt)).
        
        known_segs = []
        
        for k in range(1, N + 1):
            # Solve for segment $k$ (counting from expiry backwards).
            # Maturity for this step: k * dt ???
            # No. We add one segment at the beginning (time 0 relative to current step).
            # Current step "now" corresponds to T - k*dt in global time.
            # Maturity local: k * dt.
            # Known segments are the ones we already found (valid for t > dt local).
            # We solve for segment [0, dt] local.
            
            # Shift known segments for local time?
            # Previous segments were valid for t' in [0, ...].
            # Now we step back by dt.
            # Old segments are now [dt, ...].
            # We need to adjust parameters?
            # B(t_old) = B_old exp(b_old t_old).
            # t_new = t_old + dt.
            # B(t_new) = B_old exp(b_old (t_new - dt)) = (B_old exp(-b_old dt)) * exp(b_old t_new).
            # So shift: B -> B * exp(-b * dt). b -> same.
            # Interval [s, e] -> [s+dt, e+dt].
            
            shifted_known = []
            for seg in known_segs:
                shifted_known.append({
                    'start': seg['start'] + dt,
                    'end': seg['end'] + dt,
                    'B': seg['B'] * np.exp(-seg['b'] * dt),
                    'b': seg['b']
                })
                
            # Now solve for new segment [0, dt]
            # T_calc = k * dt ? Actually end of last segment.
            # Total maturity for this solve is k * dt.
            
            curr_maturity = k * dt
            
            # Initial guess
            if k == 1:
                guess_B, guess_b = None, None
            else:
                guess_B, guess_b = known_segs[0]['B'], known_segs[0]['b'] # Use previous head
            
            B_new, b_new = self._solve_segment(0, curr_maturity, shifted_known, guess_B, guess_b)
            
            # Add new segment at head
            new_seg = {'start': 0.0, 'end': dt, 'B': B_new, 'b': b_new}
            known_segs = [new_seg] + shifted_known
            
        # After loop, known_segs is the full boundary for T.
        # Compute Price
        
        final_segs = known_segs
        
        prem = 0
        for seg in final_segs:
            v1 = self._I_func(seg['start'], seg['end'], self.S, seg['B'], seg['b'], self.r)
            v2 = self._I_func(seg['start'], seg['end'], self.S, seg['B'], seg['b'], self.delta)
            prem += self.r * self.K * v1 - self.delta * self.S * v2
            
        return self.european_put_price(self.S, self.T) + prem

    def price(self):
        # EXP3 Method
        p1 = self.compute_price_N(1)
        p2 = self.compute_price_N(2)
        p3 = self.compute_price_N(3)
        
        return 4.5 * p3 - 4.0 * p2 + 0.5 * p1

if __name__ == "__main__":
    # Test case from Exhibit I
    # S=100, K=100, T=0.5, r=0.03, d=0.07, sigma=0.2
    # Call value -> Put(K=100, S=100, r=0.07, d=0.03) ?
    # Symmetry: Call(S, K, r, d) = Put(K, S, d, r).
    # d=0.07, r=0.03.
    # Put args: S'=100, K'=100, r'=0.07, d'=0.03.
    
    pricer = Ju1998Piecewise(100, 100, 0.03, 0.5, 0.2, 0.07, 'call')
    print(f"EXP3 Price: {pricer.price()}")
