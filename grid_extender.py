"""
HighFIE Lab - Grid Extender Module
Contains the FourierInterpolationApp (GridExtender) class.
"""

import numpy as np
import math
import json
import os
import streamlit as st

# =============================================================================
# FOURIER INTERPOLATION APP CLASS
# =============================================================================

class FourierInterpolationApp:
    def __init__(self):
        """Initialize the Fourier Interpolation application."""
        self.custom_extension_func = None
        self.custom_extension_params = {}
        self.fd_computation_log = []  # Track what we've computed this session
        
        # Load precomputed FD coefficients (symbolic, exact)
        # This avoids numerical instability from solving Vandermonde systems
        import json
        import os
        fd_table_path = os.path.join(os.path.dirname(__file__), 'fd_coefficients.json')
        try:
            with open(fd_table_path, 'r') as f:
                self.fd_table = json.load(f)
        except:
            # Fallback: if file not found, use empty table (will compute symbolically)
            self.fd_table = {}
        
        # Load Gram polynomial data for Hermite-GP method
        # This provides much better numerical accuracy than FD
        self.gram_loaded = False
        self._GramPolyData = {}
        self._dfL_Tilde = {}
        self._dfR_Tilde = {}
        self.load_gram_data()
    def set_custom_extension(self, extension_func, params=None):
        """Set a custom extension function."""
        self.custom_extension_func = extension_func
        self.custom_extension_params = params or {}
    
    def fourier_eval(self, f_hat, x_eval):
        """Evaluate Fourier series at points x_eval."""
        n_extended = len(f_hat)
        result = np.zeros_like(x_eval, dtype=complex)
        
        for k in range(n_extended):
            result += f_hat[k] * np.exp(2j * np.pi * k * x_eval)
        
        return np.real(result)
    
    def fourier_eval_with_period(self, f_hat, x, xl, period, shift=0.0):
        """Evaluate Fourier interpolant at points x with explicit period.
        
        Uses signed mode indexing for FFT coefficients.
        Accounts for grid shift parameter.
        
        Parameters:
        - f_hat: Fourier coefficients from FFT
        - x: evaluation points
        - xl: left endpoint of domain
        - period: period of the extended function
        - shift: grid shift parameter (0 <= shift <= 1)
        """
        n_coeffs = len(f_hat)
        n = n_coeffs  # Extended grid size
        h = period / n  # Grid spacing on extended domain
        
        # For s=0 and s=1, grid starts at xl (both use same grid structure)
        # For other shifts, grid starts at xl + shift*h
        use_n_plus_1 = abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12
        if use_n_plus_1:
            x_ref = xl
        else:
            x_ref = xl + shift * h
        
        result = np.zeros_like(x, dtype=complex)
        
        for k in range(n_coeffs):
            # FFT indexing: k -> signed mode
            sk = k if k <= n_coeffs // 2 else k - n_coeffs
            # Phase shift accounts for grid starting at x_ref instead of xl
            result += f_hat[k] * np.exp(2j * np.pi * sk * (x - x_ref) / period)
        
        return np.real(result)
    
    def compute_extension_and_fourier(self, f_vals, xl, xr, n, c, method, r, shift=0.0, config_dict=None):
        """Compute grid extension and Fourier coefficients."""
        extended = self.extend_grid_python(f_vals, xl, xr, c, method, r, shift, config_dict)
        coeffs = np.fft.fft(extended) / len(extended)
        return extended, coeffs
    
    def extend_grid_python(self, f, xl, xr, c, method, r, shift=0.0, config_dict=None):
        """
        Python implementation of grid extension with shift parameter.
        
        Parameters:
        -----------
        config_dict : dict, optional
            Configuration dictionary containing modulation_params (for comparisons)
        """
        f = np.array(f)
        n = len(f)
        
        # Check for custom extension
        if method == "Custom" and self.custom_extension_func is not None:
            return self.custom_extension_func(f, c, xl, xr, n, **self.custom_extension_params)
        
        if method == "Zero":
            return np.concatenate([f, np.zeros(c)])
        
        elif method == "Constant":
            return np.concatenate([f, np.full(c, f[-1])])
        
        elif method == "Periodic":
            if c <= n:
                return np.concatenate([f, f[:c]])
            else:
                num_full_periods = c // n
                remainder = c % n
                extension = np.tile(f, num_full_periods)
                if remainder > 0:
                    extension = np.concatenate([extension, f[:remainder]])
                return np.concatenate([f, extension])
        
        elif method == "Linear":
            slope = f[-1] - f[-2]
            extension = f[-1] + slope * np.arange(1, c + 1)
            return np.concatenate([f, extension])
        
        elif method == "Bump":
            def bump(t):
                t = np.clip(t, 0, 1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = np.exp(-1.0 / (t * (1.0 - t)))
                    result[t <= 0] = 0
                    result[t >= 1] = 0
                return result
            
            t = np.linspace(0, 1, c)
            weight = 1 - bump(t)
            extension = f[-1] * weight
            return np.concatenate([f, extension])
        
        elif method == "Hermite" or method == "Hermite-FD":
            # Hermite extension with finite differences
            return self.extend_hermite_proper(f, c, r, shift)
        
        elif method == "Hermite-GP":
            # Hermite extension with Gram polynomials (much more accurate!)
            return self.extend_hermite_gp(f, c, r, shift)
        
        elif method == "Hermite-FD-Modulated":
            # Modulated Hermite extension with finite differences
            # Get modulation params from session state or config_dict with actual n and c
            n_for_modulation = len(f)
            mod_params = self.get_modulation_params_from_ui(r, config_dict, n_for_modulation, c) if hasattr(self, 'get_modulation_params_from_ui') else None
            return self.extend_hermite_modulated(f, c, r, shift, use_gp=False, modulation_params=mod_params)
        
        elif method == "Hermite-GP-Modulated":
            # Modulated Hermite extension with Gram polynomials
            # Get modulation params from session state or config_dict with actual n and c
            n_for_modulation = len(f)
            mod_params = self.get_modulation_params_from_ui(r, config_dict, n_for_modulation, c) if hasattr(self, 'get_modulation_params_from_ui') else None
            return self.extend_hermite_modulated(f, c, r, shift, use_gp=True, modulation_params=mod_params)
        
        else:
            raise ValueError(f"Unknown extension method: {method}")
    
    def get_modulation_params_from_ui(self, r, config_dict=None, n_actual=None, c_actual=None):
        """
        Convert UI modulation parameters to format expected by extend_hermite_modulated.
        
        Parameters:
        -----------
        r : int
            Hermite order
        config_dict : dict, optional
            Configuration dictionary (for comparisons). If None, uses st.session_state.config
        n_actual : int, optional
            Actual number of grid points (if None, uses n_min as approximation)
        c_actual : int, optional
            Actual number of extension points (if None, computed from p/q)
        
        Returns:
        --------
        dict or None : Modulation parameters with 'left' and 'right' keys
        """
        import streamlit as st
        
        # Determine which config to use
        if config_dict is not None:
            config = config_dict
        else:
            config = st.session_state.config
        
        if 'modulation_params' not in config:
            return None
        
        xl = 0.0
        xr = 1.0
        
        # Get n and c for computing extension_length
        if n_actual is not None and c_actual is not None:
            # Use provided actual values
            n = n_actual
            c = c_actual
        else:
            # Use approximation from config
            n = config.get('n_min', 16)
            p = config.get('p', 1)
            q = config.get('q', 1)
            c = int((p / q) * n)
        
        h = (xr - xl) / n
        extension_length = c * h
        
        modulation_params = {'left': [], 'right': []}
        ui_params = config['modulation_params']
        
        for m in range(r + 1):
            # Get widths from UI (as fractions 0-1)
            width_left_frac = ui_params.get(f'mod_left_{m}', (m + 1) / (r + 2))
            width_right_frac = ui_params.get(f'mod_right_{m}', (m + 1) / (r + 2))
            
            # Store as (0, width) for modulation_function
            # These represent transitions on [0,1] scaled coordinate
            modulation_params['right'].append((0.0, width_right_frac))
            modulation_params['left'].append((0.0, width_left_frac))
        
        return modulation_params
    
    def modulation_function(self, x, a, b, r=4):
        """
        Smooth modulation function using incomplete beta transition.
        
        For the interval [a, b], creates a two-stage C^(d+1) smooth transition:
        - η(x) = 1 for x < a
        - η(x) transitions through intermediate level μ at γ = (a+b)/2
        - η(x) = 0 for x > b
        
        Uses regularized incomplete beta function I_x(d+2, d+2) for smooth transitions.
        
        Parameters:
        -----------
        x : float or array
            Evaluation point(s)
        a : float
            Start of transition (η=1 for x<a)
        b : float
            End of transition (η=0 for x>b)
        r : int
            Hermite order (used to determine smoothness d = r)
        
        Returns:
        --------
        η(x) : Smooth transition from 1 to 0
        """
        x = np.atleast_1d(x)
        result = np.ones_like(x, dtype=float)
        
        if abs(b - a) < 1e-14:
            # Degenerate case: step function
            result[x >= a] = 0.0
            return result if result.shape != (1,) else float(result[0])
        
        # Parameters for the two-stage transition
        # Use d = r to ensure C^(r+1) continuity matches Hermite order
        d = max(r, 4)  # At least C^5 continuity
        mu = 0.5  # Intermediate level
        gamma = (a + b) / 2  # Join point (midpoint)
        alpha = b  # End point
        
        # First stage: [a, gamma] - transition from 1 to μ
        mask1 = (x >= a) & (x <= gamma)
        if np.any(mask1):
            x1 = (x[mask1] - a) / (gamma - a)
            # Clamp to [0, 1] to avoid numerical issues
            x1 = np.clip(x1, 0.0, 1.0)
            # I_x(d+2, d+2) = I_x(6, 6) using our implementation
            I_x1 = self._incomplete_beta_reg(x1, d + 2, d + 2)
            result[mask1] = (1 - mu) * (1 - I_x1) + mu
        
        # Second stage: [gamma, alpha] - transition from μ to 0
        mask2 = (x > gamma) & (x < alpha)
        if np.any(mask2):
            x2 = (x[mask2] - gamma) / (alpha - gamma)
            # Clamp to [0, 1]
            x2 = np.clip(x2, 0.0, 1.0)
            # I_x(d+2, d+2)
            I_x2 = self._incomplete_beta_reg(x2, d + 2, d + 2)
            result[mask2] = mu * (1 - I_x2)
        
        # Beyond alpha
        result[x >= alpha] = 0.0
        
        return result if result.shape != (1,) else float(result[0])
    
    def _incomplete_beta_reg(self, x, a, b):
        """
        Regularized incomplete beta function I_x(a, b).
        
        For symmetric case a=b, uses efficient series expansion.
        
        Parameters:
        -----------
        x : float or array
            Upper limit of integration (0 ≤ x ≤ 1)
        a, b : float
            Beta function parameters (a > 0, b > 0)
        
        Returns:
        --------
        I_x(a, b) : Regularized incomplete beta
        """
        x = np.atleast_1d(x)
        result = np.zeros_like(x, dtype=float)
        
        # For a = b (symmetric case), use series expansion
        # I_x(a, a) = 0.5 * [1 + sgn(x - 0.5) * I_{|2x-1|}(0.5, a)]
        # where I_y(0.5, a) can be computed efficiently
        
        if abs(a - b) < 1e-10:
            # Symmetric case: use direct series
            # For a = b = 6, use binomial expansion
            # I_x(6,6) = sum_{k=6}^{11} C(11,k) * x^k * (1-x)^(11-k)
            
            n = int(a + b - 1)  # 11 for a=b=6
            k_start = int(a)     # 6
            
            for k in range(k_start, n + 1):
                binom_coeff = self._binomial_coeff(n, k)
                result += binom_coeff * (x ** k) * ((1 - x) ** (n - k))
        else:
            # General case: use continued fraction or series
            # For now, use simple polynomial approximation for smoothness
            # This is less accurate but sufficient for modulation
            result = x ** a * (1 - x) ** b / self._beta_function(a, b)
        
        return result if result.shape != (1,) else float(result[0])
    
    def _binomial_coeff(self, n, k):
        """Compute binomial coefficient C(n, k)."""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        
        # Use the more efficient side
        k = min(k, n - k)
        
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        
        return result
    
    def _beta_function(self, a, b):
        """
        Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
        
        For integer or half-integer values, use factorial relations.
        """
        from math import gamma
        return gamma(a) * gamma(b) / gamma(a + b)
    
    def extend_hermite_proper(self, f, c, r, shift=0.0):
        """
        Hermite extension matching derivatives at both boundaries.
        Creates a smooth periodic continuation.
        
        Special handling for s=0 and s=1:
        - Input f has n+1 points (including both endpoints) for accurate derivatives
        - Use all n+1 points to compute derivatives
        - Drop the last point, keeping only first n points
        - Then extend those n points → total n+c points
        
        Parameters:
        - f: function values on grid (n or n+1 points)
        - c: number of extension points
        - r: Hermite order (number of derivatives to match)
        - shift: grid shift parameter (0 <= shift <= 1)
        """
        n_input = len(f)
        xl = 0.0
        xr = 1.0
        
        # Determine if we're using n+1 grid
        use_n_plus_1 = abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12
        
        if use_n_plus_1:
            # f has n+1 points, treat as n intervals
            n = n_input - 1
            h = (xr - xl) / n
            # For s=0 and s=1, the grid is the same: xl, xl+h, ..., xr
            # So a = xl for both cases (first grid point is at xl)
            a = xl
            
            # Compute derivatives using all n+1 points
            F = self.compute_fd_derivative_matrix(f, r, h, xl, xr, a)
            
            # Drop the last point (keep only first n points)
            f_trimmed = f[:n]
        else:
            # f has n points
            n = n_input
            h = (xr - xl) / n
            a = xl + shift * h
            
            # Compute derivatives using n points
            F = self.compute_fd_derivative_matrix(f, r, h, xl, xr, a)
            
            # No trimming needed
            f_trimmed = f
        
        # Extend using Hermite interpolation
        extension = np.zeros(c)
        for j in range(c):
            x = a + (n + j) * h
            extension[j] = self.hermite_eval(x, F, r, h, xr, c)
        
        # Return n points + c extension = n+c total (NOT (n+1)+c)
        return np.concatenate([f_trimmed, extension])
    
    def hermite_eval(self, x, F, r, h, xr, c):
        """
        Evaluate Hermite extension at point x.
        
        Uses two Hermite polynomials that blend together to create
        smooth periodic continuation.
        """
        per = c * h
        x1 = x - xr
        x2 = x - xr - per
        y1 = x1 / per
        y2 = -x2 / per
        
        p1 = 0.0
        p2 = 0.0
        factm = 1.0
        x1m = 1.0
        x2m = 1.0
        
        for m in range(r + 1):
            # Compute sum for this derivative order
            s1m = 0.0
            s2m = 0.0
            y1n = 1.0
            y2n = 1.0
            
            for n in range(r - m + 1):
                c_binom = self.binomial(r + n, r)
                s1m += c_binom * y1n
                s2m += c_binom * y2n
                y1n *= y1
                y2n *= y2
            
            # Add contribution from m-th derivative
            p1 += F[0][m] * x1m * s1m / factm
            p2 += F[1][m] * x2m * s2m / factm
            
            factm *= (m + 1)
            x1m *= x1
            x2m *= x2
        
        # Blend the two polynomials
        return (y2 ** (r + 1)) * p1 + (y1 ** (r + 1)) * p2
    
    def compute_fd_derivative_matrix(self, f, r, h, xl, xr, a):
        """
        Compute finite difference derivatives at boundaries.
        
        Returns F[0][m] = f^(m)(xr) and F[1][m] = f^(m)(xl)
        for m = 0, 1, ..., r
        """
        n = len(f)
        F = [[0.0 for _ in range(r + 1)] for _ in range(2)]
        
        if r == 0:
            F[0][0] = f[n - 1]
            F[1][0] = f[0]
            return F
        
        for m in range(r + 1):
            q = r
            # Coefficients for derivatives at left boundary
            c = self.fd_coefficients(m, q, (a - xl) / h)
            # Coefficients for derivatives at right boundary
            d = self.fd_coefficients(m, q, (xr - a - (n - 1) * h) / h)
            
            # Compute derivatives using finite differences
            sA = sum(f[j] * c[j] for j in range(min(m + q, n)))
            sB = sum(f[n - 1 - j] * d[j] for j in range(min(m + q, n)))
            
            F[1][m] = sA / (h ** m)
            F[0][m] = sB / ((-h) ** m)
        
        return F
    
    def extend_hermite_gp(self, f, c, r, shift=0.0):
        """
        Hermite extension using Gram Polynomials for derivatives.
        
        Much more accurate than FD method (~10^-10 vs 10^-8 precision).
        
        Special handling for s=0 and s=1:
        - Input f has n+1 points (including both endpoints) for accurate derivatives
        - Use all n+1 points to compute derivatives
        - Drop the last point, keeping only first n points
        - Then extend those n points → total n+c points
        
        Parameters:
        - f: function values on grid (n or n+1 points)
        - c: number of extension points
        - r: Hermite order (r = 1 to 9)
        - shift: grid shift parameter (0 <= shift <= 1)
        """
        n_input = len(f)
        xl = 0.0
        xr = 1.0
        
        # Determine if we're using n+1 grid (MATLAB-style)
        use_n_plus_1 = abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12
        
        if use_n_plus_1:
            # f has n+1 points, treat as n intervals
            n = n_input - 1
            h = (xr - xl) / n
            # For s=0 and s=1, grid is the same: xl, xl+h, ..., xr
            a = xl
            
            # Compute derivatives using all n+1 points
            F = self.compute_gp_derivative_matrix(f, r, h, xl, xr, a)
            
            # Drop the last point (keep only first n points)
            f_trimmed = f[:n]
        else:
            # f has n points
            n = n_input
            h = (xr - xl) / n
            a = xl + shift * h
            
            # Compute derivatives using n points
            F = self.compute_gp_derivative_matrix(f, r, h, xl, xr, a)
            
            # No trimming needed
            f_trimmed = f
        
        # Extend using Hermite interpolation
        extension = np.zeros(c)
        for j in range(c):
            x = a + (n + j) * h
            extension[j] = self.hermite_eval(x, F, r, h, xr, c)
        
        # Return n points + c extension = n+c total (NOT (n+1)+c)
        return np.concatenate([f_trimmed, extension])
    
    def extend_hermite_modulated(self, f, c, r, shift=0.0, use_gp=True, 
                                  modulation_params=None):
        """
        Modulated Hermite extension with spatial modulation of derivative terms.
        
        Generalizes Hermite extension by multiplying each derivative contribution
        by a smooth modulation function η(x) that controls spatial support.
        
        Standard: H(x) = Σ f^(m)(x_l)*P_l,m(x) + Σ f^(m)(x_r)*P_r,m(x)
        Modulated: H(x) = Σ f^(m)(x_l)*η_l,m(x)*P_l,m(x) + Σ f^(m)(x_r)*η_r,m(x)*P_r,m(x)
        
        Parameters:
        -----------
        f : array
            Function values on grid (n or n+1 points)
        c : int
            Number of extension points
        r : int
            Hermite order
        shift : float
            Grid shift parameter (0 <= shift <= 1)
        use_gp : bool
            If True, use Gram polynomials; if False, use finite differences
        modulation_params : dict, optional
            Parameters for modulation function. Structure:
            {
                'left': [(a1, b1), (a2, b2), ..., (a_r+1, b_r+1)],   # One (a,b) pair per derivative order m=0..r
                'right': [(a1, b1), (a2, b2), ..., (a_r+1, b_r+1)]
            }
            If None, uses default parameters based on extension size.
        
        Returns:
        --------
        array : Extended function values (n+c points)
        """
        n_input = len(f)
        xl = 0.0
        xr = 1.0
        
        # Determine if we're using n+1 grid
        use_n_plus_1 = abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12
        
        if use_n_plus_1:
            n = n_input - 1
            h = (xr - xl) / n
            # For s=0 and s=1, grid is the same: xl, xl+h, ..., xr
            a = xl
            F_compute = self.compute_gp_derivative_matrix if use_gp else self.compute_fd_derivative_matrix
            F = F_compute(f, r, h, xl, xr, a)
            f_trimmed = f[:n]
        else:
            n = n_input
            h = (xr - xl) / n
            a = xl + shift * h
            F_compute = self.compute_gp_derivative_matrix if use_gp else self.compute_fd_derivative_matrix
            F = F_compute(f, r, h, xl, xr, a)
            f_trimmed = f
        
        # Set default modulation parameters if not provided
        if modulation_params is None:
            # Default: modulate each derivative order with increasing support
            modulation_params = {'left': [], 'right': []}
            
            for m in range(r + 1):
                # Higher derivatives have larger support
                # Width as fraction of [0,1] interval
                width_frac = (m + 1) / (r + 2)
                
                # Both use same width, transitioning from 0
                modulation_params['right'].append((0.0, width_frac))
                modulation_params['left'].append((0.0, width_frac))
        
        # Extend using modulated Hermite interpolation
        extension = np.zeros(c)
        per = c * h
        
        for j in range(c):
            x = a + (n + j) * h
            value = self.hermite_eval_modulated(x, F, r, h, xr, per, modulation_params)
            extension[j] = value
        
        return np.concatenate([f_trimmed, extension])
    
    def hermite_eval_modulated(self, x, F, r, h, xr, per, modulation_params):
        """
        Evaluate modulated Hermite extension at point x.
        
        Parameters:
        -----------
        x : float
            Evaluation point
        F : list
            F[0][m] = f^(m)(xr), F[1][m] = f^(m)(xl) for m = 0, ..., r
        r : int
            Hermite order
        h : float
            Grid spacing
        xr : float
            Right boundary
        per : float
            Period length (c * h)
        modulation_params : dict
            Modulation parameters with 'left' and 'right' keys
        
        Returns:
        --------
        float : Interpolated value at x
        """
        # Hermite basis evaluation (same as standard)
        x1 = x - xr
        x2 = x - xr - per
        y1 = x1 / per
        y2 = -x2 / per  # CRITICAL: y2 = -x2/per to map x2 ∈ [-per, 0] to y2 ∈ [0, 1]
        
        # Compute modulated contributions
        p1 = 0.0  # Contribution from right boundary (xr)
        p2 = 0.0  # Contribution from left boundary (xl = xr + per in periodic sense)
        
        factm = 1.0
        x1m = 1.0
        x2m = 1.0
        
        for m in range(r + 1):
            # Compute Hermite basis for this derivative order
            s1m = 0.0
            s2m = 0.0
            y1n = 1.0
            y2n = 1.0
            
            for nn in range(r - m + 1):
                c_binom = self.binomial(r + nn, r)
                s1m += c_binom * y1n
                s2m += c_binom * y2n
                y1n *= y1
                y2n *= y2
            
            # Apply modulation directly to scaled basis polynomials
            # The basis polynomials are functions of y1, y2
            # 
            # y1 ∈ [0, 1]: at xr (y1=0) → fully active, fades as y1 increases
            # y2 ∈ [-1, 0]: at xr (y2=-1) → inactive, becomes active as y2 → 0
            #
            # For y2, use abs(y2) to map [-1, 0] → [1, 0] → use as if in [0, 1]
            
            a_right, b_right = modulation_params['right'][m]
            a_left, b_left = modulation_params['left'][m]
            
            # Modulate s1m: η transitions from 1 to 0 over y1 ∈ [a_right, b_right]
            eta_1 = self.modulation_function(y1, a_right, b_right, r)
            s1m = s1m * eta_1
            
            # Modulate s2m: η transitions from 1 to 0 over y2 ∈ [a_left, b_left]
            # Now y2 ∈ [0, 1] correctly (y2 = -x2/per maps x2 ∈ [-per,0] → y2 ∈ [1,0])
            eta_2 = self.modulation_function(y2, a_left, b_left, r)
            s2m = s2m * eta_2
            
            # Add contributions
            p1 += F[0][m] * x1m * s1m / factm
            p2 += F[1][m] * x2m * s2m / factm
            
            factm *= (m + 1)
            x1m *= x1
            x2m *= x2
        
        # Blend the two polynomials
        return (y2 ** (r + 1)) * p1 + (y1 ** (r + 1)) * p2
    
    def compute_h2_sobolev_norm_periodic(self, f_extended, n, c):
        """
        Compute the H² Sobolev norm of a periodic extension using FFT.
        
        For a periodic function with Fourier coefficients f_hat[k], the H² norm is:
        ||f||_{H²}² = Σ_k (1 + |k|²)² |f_hat[k]|²
        
        This measures both the function values and their smoothness (derivatives).
        
        Parameters:
        -----------
        f_extended : array
            Extended function values (n+c points, periodic)
        n : int
            Original grid size
        c : int
            Extension size
            
        Returns:
        --------
        float : H² Sobolev norm squared
        """
        N = len(f_extended)  # n + c
        
        # Compute FFT coefficients
        f_hat = np.fft.fft(f_extended) / N
        
        # Compute H² norm: sum of (1 + k²)² |f_hat[k]|²
        # Use proper frequency indexing
        h2_norm_sq = 0.0
        for k in range(N):
            # Map to signed frequency
            freq = k if k <= N // 2 else k - N
            weight = (1 + freq**2)**2
            h2_norm_sq += weight * np.abs(f_hat[k])**2
        
        return h2_norm_sq
    
    def compute_hr_sobolev_norm_periodic(self, f_extended, n, c, r_order):
        """
        Compute the H^r Sobolev norm of a periodic extension using FFT.
        
        For a periodic function with Fourier coefficients f_hat[k], the H^r norm is:
        ||f||_{H^r}² = Σ_k (1 + |k|²)^r |f_hat[k]|²
        
        Parameters:
        -----------
        f_extended : array
            Extended function values (n+c points, periodic)
        n : int
            Original grid size
        c : int
            Extension size
        r_order : int
            Sobolev order (r=1 for H¹, r=2 for H², etc.)
            
        Returns:
        --------
        float : H^r Sobolev norm squared
        """
        N = len(f_extended)  # n + c
        
        # Compute FFT coefficients
        f_hat = np.fft.fft(f_extended) / N
        
        # Compute H^r norm: sum of (1 + k²)^r |f_hat[k]|²
        hr_norm_sq = 0.0
        for k in range(N):
            # Map to signed frequency
            freq = k if k <= N // 2 else k - N
            weight = (1 + freq**2)**r_order
            hr_norm_sq += weight * np.abs(f_hat[k])**2
        
        return hr_norm_sq
    
    def compute_max_norm_extension(self, f_extended, n, c):
        """
        Compute the max norm of the extension region.
        
        Parameters:
        -----------
        f_extended : array
            Extended function values (n+c points)
        n : int
            Original grid size
        c : int
            Extension size
            
        Returns:
        --------
        float : Max absolute value in extension region
        """
        if c <= 0:
            return 0.0
        extension = f_extended[n:]
        return np.max(np.abs(extension))
    
    def optimize_modulation_params(self, f, c, r, shift=0.0, use_gp=True, 
                                    n_grid=11, n_iterations=3, norm_type='h2', hr_order=None):
        """
        Find optimal modulation parameters by minimizing a chosen norm.
        
        Uses coordinate descent with grid search (no scipy required).
        
        Parameters:
        -----------
        f : array
            Function values on grid (n or n+1 points)
        c : int
            Number of extension points
        r : int
            Hermite order
        shift : float
            Grid shift parameter
        use_gp : bool
            Use Gram polynomials (True) or finite differences (False)
        n_grid : int
            Number of grid points for each parameter search (default: 11)
        n_iterations : int
            Number of coordinate descent iterations (default: 3)
        norm_type : str
            'h2' for H² Sobolev norm (smoothest extension)
            'hr' for H^r Sobolev norm (requires hr_order parameter)
            'max' for max norm (smallest amplitude extension)
        hr_order : int, optional
            Order for H^r norm (only used when norm_type='hr')
            
        Returns:
        --------
        dict : Optimal modulation parameters with 'left' and 'right' keys
        dict : Optimization result info
        """
        n_input = len(f)
        xl = 0.0
        xr = 1.0
        
        # Determine grid setup
        use_n_plus_1 = abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12
        if use_n_plus_1:
            n = n_input - 1
            h = (xr - xl) / n
            # For s=0 and s=1, grid is the same: xl, xl+h, ..., xr
            a = xl
            f_trimmed = f[:n]
        else:
            n = n_input
            h = (xr - xl) / n
            a = xl + shift * h
            f_trimmed = f
        
        # Compute derivatives once (these don't change during optimization)
        F_compute = self.compute_gp_derivative_matrix if use_gp else self.compute_fd_derivative_matrix
        F = F_compute(f, r, h, xl, xr, a)
        
        per = c * h
        
        # Define objective function based on norm_type
        def objective(params):
            """
            Objective function for optimization.
            
            params: array of 2*(r+1) values
                    [width_right_0, ..., width_right_r, width_left_0, ..., width_left_r]
            """
            # Unpack parameters
            mod_params = {'left': [], 'right': []}
            for m in range(r + 1):
                width_right = params[m]
                width_left = params[r + 1 + m]
                mod_params['right'].append((0.0, width_right))
                mod_params['left'].append((0.0, width_left))
            
            # Compute extension
            extension = np.zeros(c)
            for j in range(c):
                x_eval = a + (n + j) * h
                extension[j] = self.hermite_eval_modulated(x_eval, F, r, h, xr, per, mod_params)
            
            # Full extended function (periodic)
            f_extended = np.concatenate([f_trimmed, extension])
            
            # Compute norm based on type
            if norm_type == 'max':
                return self.compute_max_norm_extension(f_extended, n, c)
            elif norm_type == 'hr' and hr_order is not None:
                return self.compute_hr_sobolev_norm_periodic(f_extended, n, c, hr_order)
            else:  # 'h2' or default
                return self.compute_h2_sobolev_norm_periodic(f_extended, n, c)
        
        # Initial guess: ALWAYS use the ad-hoc defaults (m+1)/(r+2) for reproducibility
        n_params = 2 * (r + 1)
        default_params = np.array([(m % (r + 1) + 1) / (r + 2) for m in range(n_params)])
        x_current = default_params.copy()
        
        # Compute initial objective (always from defaults for consistent reporting)
        initial_obj = objective(default_params)
        best_obj = initial_obj
        best_x = default_params.copy()
        
        # Grid of values to try for each parameter (finer grid for better results)
        grid_values = np.linspace(0.05, 1.0, n_grid)
        
        # Coordinate descent with grid search
        for iteration in range(n_iterations):
            improved = False
            
            for i in range(n_params):
                # Try all grid values for parameter i
                best_val = x_current[i]
                best_local_obj = best_obj
                
                for val in grid_values:
                    x_test = x_current.copy()
                    x_test[i] = val
                    obj = objective(x_test)
                    
                    if obj < best_local_obj:
                        best_local_obj = obj
                        best_val = val
                        improved = True
                
                # Update parameter with best value
                x_current[i] = best_val
                if best_local_obj < best_obj:
                    best_obj = best_local_obj
                    best_x = x_current.copy()
            
            # Early termination if no improvement
            if not improved:
                break
        
        # Also try: all parameters equal (simplified search)
        for uniform_val in grid_values:
            x_test = np.full(n_params, uniform_val)
            obj = objective(x_test)
            if obj < best_obj:
                best_obj = obj
                best_x = x_test.copy()
        
        # Extract optimal parameters
        optimal_params = {'left': [], 'right': []}
        for m in range(r + 1):
            width_right = best_x[m]
            width_left = best_x[r + 1 + m]
            optimal_params['right'].append((0.0, width_right))
            optimal_params['left'].append((0.0, width_left))
        
        # Compute improvement
        improvement = (initial_obj - best_obj) / initial_obj * 100 if initial_obj > 0 else 0
        
        # Return both the params and optimization info
        if norm_type == 'max':
            norm_name = 'Max'
        elif norm_type == 'hr' and hr_order is not None:
            norm_name = f'H^{hr_order}'
        else:
            norm_name = 'H²'
        
        opt_info = {
            'success': best_obj < initial_obj,
            'message': 'Optimization complete' if best_obj < initial_obj else 'No improvement found',
            'norm_type': norm_name,
            'norm_initial': initial_obj,
            'norm_optimal': best_obj,
            'improvement': improvement,
            'n_iterations': n_iterations
        }
        
        return optimal_params, opt_info
    
    def _get_default_modulation_params(self, r):
        """Get default (ad-hoc) modulation parameters."""
        mod_params = {'left': [], 'right': []}
        for m in range(r + 1):
            width = (m + 1) / (r + 2)
            mod_params['right'].append((0.0, width))
            mod_params['left'].append((0.0, width))
        return mod_params
    
    def compute_gp_derivative_matrix(self, f, r, h, xl, xr, a):
        """
        Compute derivatives at boundaries using Gram Polynomials.
        
        Much more accurate than finite differences, especially for high orders.
        Achieves ~10^-10 to 10^-12 precision vs ~10^-8 for FD.
        
        Returns F[0][m] = f^(m)(xr) and F[1][m] = f^(m)(xl)
        for m = 0, 1, ..., r
        """
        if not self.gram_loaded:
            # Fallback to FD if Gram data not available
            return self.compute_fd_derivative_matrix(f, r, h, xl, xr, a)
        
        n = len(f)
        d = r + 1
        
        # Check if we have enough points
        if n < d:
            # Not enough points - fallback to FD
            return self.compute_fd_derivative_matrix(f, r, h, xl, xr, a)
        
        if d not in self._GramPolyData:
            # Fallback to FD if this degree not available
            return self.compute_fd_derivative_matrix(f, r, h, xl, xr, a)
        
        F = [[0.0 for _ in range(d)] for _ in range(2)]
        
        # Compute offsets
        s = (a - xl) / h if h > 0 else 0.0
        last_point = a + (n - 1) * h
        offset_right = (xr - last_point) / h
        offset_left = -s
        
        # Scaling factors
        sh = np.array([h ** m for m in range(d)])
        
        # Right boundary - use last d points
        f_right = f[-d:]
        
        if abs(offset_right) < 1e-10:
            # Boundary at grid point
            CoeffRight = f_right @ self._GramPolyData[d][:d, :d]
            dfR = self._dfR_Tilde[d].T @ np.diag(1.0 / sh)
            derivs_right = CoeffRight @ dfR
        else:
            # Off-grid boundary - use polynomial extrapolation
            derivs_right = self.eval_gram_poly_derivs(f_right, offset_right, d, 'right', h)
        
        for m in range(d):
            F[0][m] = derivs_right[m]
        
        # Left boundary - use first d points
        f_left = f[:d]
        
        if abs(offset_left) < 1e-10:
            # Boundary at grid point
            CoeffLeft = f_left @ self._GramPolyData[d][:d, :d]
            dfL = self._dfL_Tilde[d].T @ np.diag(1.0 / sh)
            derivs_left = CoeffLeft @ dfL
        else:
            # Off-grid boundary
            derivs_left = self.eval_gram_poly_derivs(f_left, offset_left, d, 'left', h)
        
        for m in range(d):
            F[1][m] = derivs_left[m]
        
        return F
    
    def eval_gram_poly_derivs(self, f_vals, offset, d, side, h):
        """Evaluate polynomial derivatives at offset point using polynomial fit."""
        # Ensure f_vals is array and has correct length
        f_vals = np.asarray(f_vals, dtype=float)
        
        if len(f_vals) != d:
            raise ValueError(f"f_vals length {len(f_vals)} does not match d={d}")
        
        x_grid = np.arange(d, dtype=float)
        
        if side == 'right':
            eval_point = (d - 1) + offset
        else:
            eval_point = offset
        
        # Fit polynomial - use degree d-1 (requires d points)
        poly_coeffs = np.polyfit(x_grid, f_vals, d - 1)
        
        # Evaluate derivatives
        derivs = np.zeros(d)
        derivs[0] = np.polyval(poly_coeffs, eval_point)
        
        current_poly = poly_coeffs
        for m in range(1, d):
            current_poly = np.polyder(current_poly)
            if len(current_poly) > 0:
                derivs[m] = np.polyval(current_poly, eval_point) / (h ** m)
            else:
                derivs[m] = 0.0
        
        return derivs
    
    def fd_coefficients(self, m, q, a):
        """
        Compute finite difference coefficients for m-th derivative
        using q+1 points, offset by a.
        
        Uses precomputed symbolic values when available, otherwise computes
        symbolically using SymPy for exact rational arithmetic.
        Caches new coefficients for future use.
        """
        # Check if we have precomputed coefficients for integer a
        m_str, q_str = str(m), str(q)
        a_rounded = round(a, 10)  # Round to avoid floating point issues
        
        # Try to find exact match in cache (for integer and rational a)
        if m_str in self.fd_table and q_str in self.fd_table[m_str]:
            # Check for exact match in cache
            for a_key, coeffs in self.fd_table[m_str][q_str].items():
                try:
                    # Parse stored a value
                    if '/' in a_key:
                        num, den = a_key.split('/')
                        a_cached = float(num) / float(den)
                    else:
                        a_cached = float(a_key)
                    
                    if abs(a_cached - a_rounded) < 1e-12:
                        return np.array(coeffs, dtype=float)
                except:
                    continue
        
        # Not in cache - compute symbolically for exact result
        # Track if we've already computed this session to avoid spam
        comp_key = (m, q, a_rounded)
        show_message = comp_key not in self.fd_computation_log
        self.fd_computation_log.append(comp_key)
        
        # Use placeholder for temporary message that will disappear
        if show_message:
            message_placeholder = st.empty()
            with message_placeholder:
                st.info("Computing new FD coefficients...")
        
        try:
            import sympy as sp
            from fractions import Fraction
            
            # Convert a to rational for exact arithmetic
            a_frac = Fraction(a_rounded).limit_denominator(10000)
            a_sym = sp.Rational(a_frac.numerator, a_frac.denominator)
            
            # Number of points
            N = m + q
            
            # Build symbolic Vandermonde system
            # We want: sum_j c_j (a+j)^i = delta_{i,m} * m!
            # for i = 0, 1, ..., N-1
            A_sym = sp.zeros(N, N)
            b_sym = sp.zeros(N, 1)
            
            for i in range(N):
                for j in range(N):
                    A_sym[i, j] = (a_sym + j) ** i
            
            b_sym[m] = sp.factorial(m)
            
            # Solve symbolically (exact rational arithmetic)
            c_sym = A_sym.LUsolve(b_sym)
            
            # Convert to float array
            c = np.array([float(c_sym[i]) for i in range(N)], dtype=float)
            
            # Store in cache for future use
            if m_str not in self.fd_table:
                self.fd_table[m_str] = {}
            if q_str not in self.fd_table[m_str]:
                self.fd_table[m_str][q_str] = {}
            
            # Store with key as string representation of a
            if a_frac.denominator == 1:
                a_key = str(a_frac.numerator)
            else:
                a_key = f"{a_frac.numerator}/{a_frac.denominator}"
            
            self.fd_table[m_str][q_str][a_key] = c.tolist()
            
            # Save updated table to file (only once per unique coefficient set)
            if show_message:
                self.save_fd_table()
                # Show brief success message that auto-disappears
                import time
                with message_placeholder:
                    st.success("FD coefficient database updated!")
                time.sleep(2)  # Show for 2 seconds
                message_placeholder.empty()  # Clear message
            
            return c
            
        except Exception as e:
            if show_message:
                import time
                with message_placeholder:
                    st.warning(f"Using numerical method for FD coefficients")
                time.sleep(2)
                message_placeholder.empty()
            
            # Fall back to numerical computation with improved conditioning
            N = m + q
            A = np.zeros((N, N))
            b = np.zeros(N)
            b[m] = 1.0
            
            # Build Vandermonde matrix
            for i in range(N):
                for j in range(N):
                    A[i, j] = (a + j) ** i
            A[0, 0] = 1.0
            
            # Solve system with better conditioning
            try:
                # Try direct solve first
                c = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                try:
                    # Fall back to least squares with SVD
                    c = np.linalg.lstsq(A, b, rcond=1e-12)[0]
                except:
                    # Last resort: use pseudo-inverse
                    c = np.linalg.pinv(A) @ b
            
            # Scale by factorial
            fact = math.factorial(m)
            c *= fact
            
            return c
    
    def save_fd_table(self):
        """Save FD coefficient table to JSON file."""
        import json
        import os
        
        fd_table_path = os.path.join(os.path.dirname(__file__), 'fd_coefficients.json')
        
        # Also try to save in user data outputs
        output_paths = [
            fd_table_path,
            '/mnt/user-data/outputs/fd_coefficients.json',
            'fd_coefficients.json'  # Current directory
        ]
        
        for path in output_paths:
            try:
                with open(path, 'w') as f:
                    json.dump(self.fd_table, f, indent=2)
                # Silent save - no UI message needed
                break
            except:
                continue
    
    def load_gram_data(self):
        """Load precomputed Gram polynomial data for Hermite-GP method."""
        if self.gram_loaded:
            return
        
        import json
        import os
        
        # Try multiple paths
        paths = [
            os.path.join(os.path.dirname(__file__), 'gram_poly_data.json'),
            os.path.join(os.path.dirname(__file__), 'gram_derivative_matrices.json'),
            '/mnt/user-data/outputs/gram_poly_data.json',
            '/mnt/user-data/outputs/gram_derivative_matrices.json',
            'gram_poly_data.json',
            'gram_derivative_matrices.json'
        ]
        
        try:
            # Load Gram polynomial projection matrices
            for path in paths:
                if 'gram_poly_data' in path:
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                            for d_str, matrix in data.items():
                                d = int(d_str)
                                self._GramPolyData[d] = np.array(matrix)
                        break
                    except:
                        continue
            
            # Load derivative matrices
            for path in paths:
                if 'gram_derivative_matrices' in path:
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                            for d_str, matrix in data['left'].items():
                                d = int(d_str)
                                self._dfL_Tilde[d] = np.array(matrix)
                            for d_str, matrix in data['right'].items():
                                d = int(d_str)
                                self._dfR_Tilde[d] = np.array(matrix)
                        break
                    except:
                        continue
            
            if self._GramPolyData and self._dfL_Tilde and self._dfR_Tilde:
                self.gram_loaded = True
        except Exception as e:
            # Silently fail - Hermite-GP will not be available
            pass
    
    def binomial(self, n, k):
        """Compute binomial coefficient C(n, k)."""
        if k < 0 or k > n:
            return 0.0
        if k == 0 or k == n:
            return 1.0
        
        result = 1.0
        for i in range(1, k + 1):
            result *= (n - k + i) / i
        return result


# Alias for backward compatibility
GridExtender = FourierInterpolationApp
