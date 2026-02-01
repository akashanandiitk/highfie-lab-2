"""
HighFIE Lab - Oscillatory Integration Module
Computes high-frequency integrals: ∫_{x_l}^{x_r} w(x) f(x) e^{iωx} dx

Uses mpmath for accurate evaluation of special functions (incomplete gamma,
confluent hypergeometric, Bessel functions) with complex arguments.

Includes a moment cache to avoid recomputing moments for the same configuration.
"""

import numpy as np
from scipy.special import gamma as gamma_func
from scipy import integrate
import json
import os

try:
    import mpmath
    from mpmath import mp, mpf, mpc, gammainc, hyp1f1, gamma as mp_gamma
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False
    print("Warning: mpmath not available. Some moment calculations may be inaccurate.")


# Default cache file location (next to this module)
_DEFAULT_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "moment_cache.json")


# ==========================================================================
# MOMENT CACHE
# ==========================================================================

class MomentCache:
    """
    Cache for precomputed moments W_{ω,k}.
    
    Key: (xl, xr, omega, k, n, c, weight_type, alpha_w, beta_w)
    Value: complex moment W_{ω,k}
    
    Supports JSON persistence so moments survive across sessions.
    """
    
    def __init__(self, filepath=None):
        self._cache = {}
        self._filepath = filepath or _DEFAULT_CACHE_FILE
        self._dirty = False  # Track unsaved changes
        self._load()
    
    def _make_key(self, xl, xr, omega, k, n, c, weight_type, alpha_w, beta_w):
        return (round(xl, 14), round(xr, 14), round(omega, 14), k, n, c,
                weight_type, round(alpha_w, 14), round(beta_w, 14))
    
    def _key_to_str(self, key):
        """Convert tuple key to JSON-safe string."""
        return json.dumps(list(key))
    
    def _str_to_key(self, s):
        """Convert JSON string back to tuple key."""
        lst = json.loads(s)
        return tuple(lst)
    
    def get(self, xl, xr, omega, k, n, c, weight_type, alpha_w, beta_w):
        key = self._make_key(xl, xr, omega, k, n, c, weight_type, alpha_w, beta_w)
        return self._cache.get(key, None)
    
    def put(self, xl, xr, omega, k, n, c, weight_type, alpha_w, beta_w, value):
        key = self._make_key(xl, xr, omega, k, n, c, weight_type, alpha_w, beta_w)
        self._cache[key] = value
        self._dirty = True
    
    def size(self):
        return len(self._cache)
    
    def clear(self):
        self._cache.clear()
        self._dirty = True
        self.save()
    
    def has_unsaved(self):
        return self._dirty
    
    def _load(self):
        """Load cache from JSON file if it exists."""
        if os.path.exists(self._filepath):
            try:
                with open(self._filepath, 'r') as f:
                    data = json.load(f)
                for key_str, val in data.items():
                    key = self._str_to_key(key_str)
                    self._cache[key] = complex(val[0], val[1])
            except Exception:
                pass  # Silently skip if file is corrupted
    
    def save(self):
        """Save cache to JSON file."""
        try:
            data = {}
            for key, val in self._cache.items():
                data[self._key_to_str(key)] = [val.real, val.imag]
            with open(self._filepath, 'w') as f:
                json.dump(data, f)
            self._dirty = False
            return True
        except Exception:
            return False


# Global moment cache (persists across calls and sessions via JSON)
_moment_cache = MomentCache()


def get_moment_cache():
    """Return the global moment cache."""
    return _moment_cache


# ==========================================================================
# GAMMA_K COMPUTATION
# ==========================================================================

def compute_gamma_k(a: float, b: float, omega: float, k: int, n: int, c: int) -> float:
    """
    Compute γ_k = (b-a)ω + 2πkn/(n+c)
    """
    L = b - a
    return L * omega + 2 * np.pi * k * n / (n + c)


# ==========================================================================
# MOMENT FUNCTIONS
# ==========================================================================

def compute_moment_no_weight(a: float, b: float, omega: float, k: int,
                              n: int, c: int) -> complex:
    """
    Moment for w(x) = 1 (no singularity):
    
    W_{ω,k} = e^{iωa} (b-a) ∫_0^1 e^{iγ_k u} du
    
    Result:
    - If γ_k = 0: W = e^{iωa} (b-a)
    - Otherwise: W = e^{iωa} (2(b-a)/γ_k) e^{iγ_k/2} sin(γ_k/2)
    """
    L = b - a
    gamma_k = compute_gamma_k(a, b, omega, k, n, c)
    phase = np.exp(1j * omega * a)
    
    if abs(gamma_k) < 1e-14:
        return phase * L
    
    half_gamma = gamma_k / 2
    return phase * (2 * L / gamma_k) * np.exp(1j * half_gamma) * np.sin(half_gamma)


def compute_moment_left_singularity(a: float, b: float, omega: float, k: int,
                                     n: int, c: int, beta: float) -> complex:
    """
    Moment for w(x) = (x-a)^β (left singularity), β > -1:
    
    W_{ω,k} = e^{iωa} (b-a)^{1+β} ∫_0^1 u^β e^{iγ_k u} du
    
    Key integral:
    ∫_0^1 u^β e^{iγu} du = (i/γ)^{β+1} γ(β+1, -iγ)
    
    where γ(a,z) is the lower incomplete gamma function.
    """
    L = b - a
    gamma_k = compute_gamma_k(a, b, omega, k, n, c)
    phase = np.exp(1j * omega * a)
    alpha = 1 + beta
    
    if abs(gamma_k) < 1e-14:
        return phase * L ** alpha / alpha
    
    if MPMATH_AVAILABLE:
        mp.dps = 30
        
        gamma_complete = mp_gamma(alpha)
        
        if gamma_k > 0:
            z = mpc(0, -gamma_k)  # -iγ
            gamma_upper = gammainc(alpha, z, regularized=False)
            gamma_lower = gamma_complete - gamma_upper
            i_power = mpc(0, 1) ** alpha  # i^{β+1}
            prefactor = i_power / (mpf(gamma_k) ** alpha)
        else:
            z = mpc(0, abs(gamma_k))  # i|γ| = -i(γ)  since γ<0
            gamma_upper = gammainc(alpha, z, regularized=False)
            gamma_lower = gamma_complete - gamma_upper
            prefactor = mpc(0, -1) ** alpha / (mpf(abs(gamma_k)) ** alpha)
        
        result = L ** alpha * prefactor * gamma_lower
        return phase * complex(float(result.real), float(result.imag))
    else:
        return _moment_left_series(a, b, omega, k, n, c, beta)


def compute_moment_right_singularity(a: float, b: float, omega: float, k: int,
                                      n: int, c: int, beta: float) -> complex:
    """
    Moment for w(x) = (b-x)^β (right singularity), β > -1:
    
    W_{ω,k} = e^{iωa} (b-a)^{1+β} e^{iγ_k} ∫_0^1 t^β e^{-iγ_k t} dt
    
    Key integral:
    ∫_0^1 t^β e^{-iγt} dt = (-i/γ)^{β+1} γ(β+1, iγ)
    """
    L = b - a
    gamma_k = compute_gamma_k(a, b, omega, k, n, c)
    phase = np.exp(1j * omega * a)
    alpha = 1 + beta
    
    if abs(gamma_k) < 1e-14:
        return phase * L ** alpha / alpha
    
    if MPMATH_AVAILABLE:
        mp.dps = 30
        
        gamma_complete = mp_gamma(alpha)
        
        if gamma_k > 0:
            z = mpc(0, gamma_k)  # iγ
            gamma_upper = gammainc(alpha, z, regularized=False)
            gamma_lower = gamma_complete - gamma_upper
            neg_i_power = mpc(0, -1) ** alpha
            prefactor = neg_i_power / (mpf(gamma_k) ** alpha)
        else:
            z = mpc(0, -abs(gamma_k))  # -i|γ|
            gamma_upper = gammainc(alpha, z, regularized=False)
            gamma_lower = gamma_complete - gamma_upper
            prefactor = mpc(0, 1) ** alpha / (mpf(abs(gamma_k)) ** alpha)
        
        exp_factor = mp.exp(mpc(0, gamma_k))
        result = L ** alpha * exp_factor * prefactor * gamma_lower
        return phase * complex(float(result.real), float(result.imag))
    else:
        return _moment_right_series(a, b, omega, k, n, c, beta)


def compute_moment_symmetric_singularity(a: float, b: float, omega: float, k: int,
                                          n: int, c: int, beta: float) -> complex:
    """
    Moment for symmetric weight w(x) = [(x-a)(b-x)]^β, β > -1.
    
    This is the special case of the Jacobi weight with α = β.
    Delegates to the Jacobi ₁F₁ formula which is numerically robust
    for all signs of γ_k.
    """
    return compute_moment_jacobi_weight(a, b, omega, k, n, c,
                                         alpha_w=beta, beta_w=beta)


def compute_moment_jacobi_weight(a: float, b: float, omega: float, k: int,
                                  n: int, c: int, alpha_w: float, beta_w: float) -> complex:
    """
    Moment for Jacobi weight w(x) = (x-a)^α (b-x)^β.
    Uses confluent hypergeometric function ₁F₁.
    """
    L = b - a
    gamma_k = compute_gamma_k(a, b, omega, k, n, c)
    phase = np.exp(1j * omega * a)
    
    prefactor = (L ** (alpha_w + beta_w + 1) * gamma_func(1 + alpha_w) * gamma_func(1 + beta_w)
                 / gamma_func(2 + alpha_w + beta_w))
    
    if MPMATH_AVAILABLE:
        mp.dps = 30
        z = mpc(0, gamma_k)
        hyp_val = hyp1f1(1 + alpha_w, 2 + alpha_w + beta_w, z)
        return phase * prefactor * complex(float(hyp_val.real), float(hyp_val.imag))
    else:
        z = 1j * gamma_k
        hyp_val = _hyp1f1_series(1 + alpha_w, 2 + alpha_w + beta_w, z)
        return phase * prefactor * hyp_val


# ==========================================================================
# FALLBACK SERIES EXPANSIONS
# ==========================================================================

def _moment_left_series(a, b, omega, k, n, c, beta):
    """Fallback series for left singularity moment."""
    L = b - a
    gamma_k = compute_gamma_k(a, b, omega, k, n, c)
    phase = np.exp(1j * omega * a)
    alpha = 1 + beta
    z = 1j * gamma_k
    term = 1.0 / alpha
    total = term
    for nn in range(1, 500):
        term *= z / nn / (alpha + nn)
        total += term
        if abs(term) < 1e-15 * abs(total):
            break
    return phase * L ** alpha * total


def _moment_right_series(a, b, omega, k, n, c, beta):
    """Fallback series for right singularity moment."""
    L = b - a
    gamma_k = compute_gamma_k(a, b, omega, k, n, c)
    phase = np.exp(1j * omega * a)
    alpha = 1 + beta
    z = -1j * gamma_k
    term = 1.0 / alpha
    total = term
    for nn in range(1, 500):
        term *= z / nn / (alpha + nn)
        total += term
        if abs(term) < 1e-15 * abs(total):
            break
    return phase * L ** alpha * np.exp(1j * gamma_k) * total


def _hyp1f1_series(a, b, z, max_terms=500):
    """Fallback series for ₁F₁(a, b, z)."""
    if abs(z) < 1e-14:
        return complex(1.0, 0.0)
    term = 1.0 + 0.0j
    total = term
    for nn in range(1, max_terms):
        term *= (a + nn - 1) / (b + nn - 1) * z / nn
        total += term
        if abs(term) < 1e-15 * max(abs(total), 1e-300):
            break
    return total


# ==========================================================================
# CACHED MOMENT DISPATCH
# ==========================================================================

def _compute_moment(a, b, omega, k, n, c, weight_type, alpha_w, beta_w, shift=0.0):
    """Compute a single moment, using cache if available.
    
    With non-zero shift s, the Fourier basis is
        e^{2πik(x - a - s·h) / ((n+c)·h)}
    so the moment picks up a phase factor  e^{-2πiks/(n+c)}.
    We cache the shift=0 moment and multiply by this phase.
    """
    cache = get_moment_cache()
    
    # Cache the base moment (shift = 0)
    cached = cache.get(a, b, omega, k, n, c, weight_type, alpha_w, beta_w)
    if cached is None:
        if weight_type == 'none':
            cached = compute_moment_no_weight(a, b, omega, k, n, c)
        elif weight_type == 'left':
            cached = compute_moment_left_singularity(a, b, omega, k, n, c, beta_w)
        elif weight_type == 'right':
            cached = compute_moment_right_singularity(a, b, omega, k, n, c, beta_w)
        elif weight_type == 'symmetric':
            cached = compute_moment_symmetric_singularity(a, b, omega, k, n, c, beta_w)
        elif weight_type == 'jacobi':
            cached = compute_moment_jacobi_weight(a, b, omega, k, n, c, alpha_w, beta_w)
        else:
            cached = compute_moment_no_weight(a, b, omega, k, n, c)
        cache.put(a, b, omega, k, n, c, weight_type, alpha_w, beta_w, cached)
    
    # Apply shift phase: e^{-2πi k s / (n+c)}
    if abs(shift) > 1e-14:
        N = n + c
        shift_phase = np.exp(-2j * np.pi * k * shift / N)
        return cached * shift_phase
    return cached


def precompute_moments(a, b, omega, n, c, weight_type, alpha_w, beta_w, shift=0.0):
    """
    Precompute and cache all moments for a given (omega, n, c) configuration.
    Returns the array of moments (with shift phase applied if shift != 0).
    """
    N = n + c
    moments = np.zeros(N, dtype=complex)
    for idx in range(N):
        k = idx if idx <= N // 2 else idx - N
        moments[idx] = _compute_moment(a, b, omega, k, n, c,
                                        weight_type, alpha_w, beta_w, shift)
    return moments


# ==========================================================================
# OSCILLATORY INTEGRAL (uses cache)
# ==========================================================================

def oscillatory_integral(f_coeffs: np.ndarray, a: float, b: float,
                          omega: float, n: int, c: int,
                          weight_type: str = 'none',
                          alpha_w: float = 0.0, beta_w: float = 0.0,
                          shift: float = 0.0) -> complex:
    """
    Compute I = ∫_{x_l}^{x_r} w(x) f(x) e^{iωx} dx ≈ Σ_k c_k · W_{ω,k}
    
    With non-zero shift s, the FC basis functions are
        φ_k(x) = e^{2πik(x - x_l - s·h)/((n+c)·h)}
    which introduces a phase factor e^{-2πiks/(n+c)} into each moment.
    
    Uses the moment cache to avoid redundant evaluations.
    """
    N = n + c
    moments = precompute_moments(a, b, omega, n, c, weight_type, alpha_w, beta_w, shift)
    return np.dot(f_coeffs[:N], moments)


# ==========================================================================
# REFERENCE INTEGRALS
# ==========================================================================

def compute_reference_integral(func, a, b, omega, weight_type='none',
                                alpha_w=0.0, beta_w=0.0):
    """Compute reference integral using scipy quadrature."""
    def weight_func(x):
        w = 1.0
        if weight_type == 'left' and x > a:
            w = (x - a) ** beta_w
        elif weight_type == 'right' and x < b:
            w = (b - x) ** beta_w
        elif weight_type == 'symmetric' and a < x < b:
            w = ((x - a) * (b - x)) ** beta_w
        elif weight_type == 'jacobi' and a < x < b:
            w = (x - a) ** alpha_w * (b - x) ** beta_w
        return w
    
    def integrand_real(x):
        return weight_func(x) * func(x) * np.cos(omega * x)
    
    def integrand_imag(x):
        return weight_func(x) * func(x) * np.sin(omega * x)
    
    n_oscillations = abs(omega) * (b - a) / (2 * np.pi)
    limit = max(500, int(50 * n_oscillations))
    limit = min(limit, 10000)
    
    try:
        real_part, _ = integrate.quad(integrand_real, a, b, limit=limit)
        imag_part, _ = integrate.quad(integrand_imag, a, b, limit=limit)
    except:
        n_pts = max(10000, int(100 * n_oscillations))
        n_pts = min(n_pts, 100000)
        x_pts = np.linspace(a + 1e-12, b - 1e-12, n_pts)
        real_vals = np.array([integrand_real(xi) for xi in x_pts])
        imag_vals = np.array([integrand_imag(xi) for xi in x_pts])
        real_part = np.trapezoid(real_vals, x_pts)
        imag_part = np.trapezoid(imag_vals, x_pts)
    
    return complex(real_part, imag_part)


def compute_reference_integral_mpmath(func_str, a, b, omega,
                                       weight_type='none',
                                       alpha_w=0.0, beta_w=0.0,
                                       dps=50):
    """Compute reference integral using mpmath high-precision arithmetic."""
    if not MPMATH_AVAILABLE:
        raise ImportError("mpmath is required for high-precision integration.")
    
    mp.dps = dps
    a_mp, b_mp = mpf(str(a)), mpf(str(b))
    omega_mp = mpf(str(omega))
    alpha_mp, beta_mp = mpf(str(alpha_w)), mpf(str(beta_w))
    
    def parse_func_mpmath(expr_str):
        import re
        expr = expr_str
        expr = re.sub(r'\bsin\b', 'mp.sin', expr)
        expr = re.sub(r'\bcos\b', 'mp.cos', expr)
        expr = re.sub(r'\btan\b', 'mp.tan', expr)
        expr = re.sub(r'\bexp\b', 'mp.exp', expr)
        expr = re.sub(r'\blog\b', 'mp.log', expr)
        expr = re.sub(r'\bsqrt\b', 'mp.sqrt', expr)
        expr = re.sub(r'\bpi\b', 'mp.pi', expr)
        expr = re.sub(r'\babs\b', 'mp.fabs', expr)
        expr = expr.replace('^', '**')
        return expr
    
    func_mp_str = parse_func_mpmath(func_str)
    
    def f_mpmath(x):
        try:
            return eval(func_mp_str, {'mp': mp, 'x': x, 'mpf': mpf})
        except:
            return mpf(0)
    
    def weight_mpmath(x):
        if weight_type == 'none':
            return mpf(1)
        elif weight_type == 'left':
            return (x - a_mp) ** beta_mp if x > a_mp else mpf(0)
        elif weight_type == 'right':
            return (b_mp - x) ** beta_mp if x < b_mp else mpf(0)
        elif weight_type == 'symmetric':
            return ((x - a_mp) * (b_mp - x)) ** beta_mp if a_mp < x < b_mp else mpf(0)
        elif weight_type == 'jacobi':
            return ((x - a_mp) ** alpha_mp * (b_mp - x) ** beta_mp
                    if a_mp < x < b_mp else mpf(0))
        return mpf(1)
    
    def integrand_real(x):
        return weight_mpmath(x) * f_mpmath(x) * mp.cos(omega_mp * x)
    
    def integrand_imag(x):
        return weight_mpmath(x) * f_mpmath(x) * mp.sin(omega_mp * x)
    
    try:
        real_part = mp.quad(integrand_real, [a_mp, b_mp])
        imag_part = mp.quad(integrand_imag, [a_mp, b_mp])
    except:
        real_part = mp.quad(integrand_real, [a_mp, b_mp], maxdegree=10)
        imag_part = mp.quad(integrand_imag, [a_mp, b_mp], maxdegree=10)
    
    return complex(float(real_part), float(imag_part))
