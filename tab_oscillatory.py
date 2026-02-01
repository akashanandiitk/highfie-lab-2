"""
HighFIE Lab - Oscillatory Integration Tab
Tab 3: Compute ∫_{x_l}^{x_r} w(x) f(x) e^{iωx} dx
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
from sympy import sympify, lambdify
import io

from grid_extender import FourierInterpolationApp, GridExtender
from oscillatory import (oscillatory_integral, compute_reference_integral,
                          compute_reference_integral_mpmath, get_moment_cache)


def create_download_button(fig, filename, label="Download Plot (PNG, 300 DPI)", key=None):
    """Create a download button for a matplotlib figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    st.download_button(
        label=label,
        data=buf,
        file_name=f"{filename}.png",
        mime="image/png",
        key=key,
        use_container_width=True
    )


def oscillatory_integration_tab(app):
    """Tab 3: Oscillatory Integration."""
    
    st.markdown("## Oscillatory Integration")
    st.markdown(r"""
    Compute high-frequency integrals:
    $I = \int_{x_\ell}^{x_r} w(x)\, f(x)\, e^{i\omega x}\, dx$
    using Fourier-based quadrature with FC extension.
    """)
    
    # Check if configuration exists from Tab 1
    if 'config' not in st.session_state:
        st.warning("Please configure extension parameters in the **Setup & Test** tab first.")
        st.info("The oscillatory integration uses the same function f(x) and extension settings from Tab 1.")
        return
    
    config = st.session_state.config
    params = st.session_state.get('analysis_params', {})
    
    # Show current config from Tab 1
    with st.expander("Current Configuration (from Setup Tab)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Function:** `{config.get('func_str', 'Not set')}`")
            xl_val = params.get('xl', 0)
            xr_val = params.get('xr', 1)
            st.markdown(rf"**Domain:** $[x_\ell, x_r] = [{xl_val}, {xr_val}]$")
            st.markdown(f"**Extension Method:** {config.get('method', 'Hermite-GP')}")
        with col2:
            st.markdown(f"**Hermite Order:** r = {config.get('r', 4)}")
            st.markdown(f"**Extension Ratio:** p/q = {config.get('p', 1)}/{config.get('q', 1)}")
            st.markdown(f"**Grid Shift:** s = {config.get('shift', 0.0)}")
    
    st.markdown("---")
    
    # ==========================================================================
    # WEIGHT FUNCTION CONFIGURATION
    # ==========================================================================
    st.markdown("### Weight Function")
    
    col_w1, col_w2 = st.columns([1, 1])
    
    with col_w1:
        weight_type = st.selectbox(
            "Weight w(x)",
            ['none', 'left', 'right', 'jacobi'],
            format_func=lambda x: {
                'none': 'w(x) = 1  (no weight)',
                'left': 'w(x) = (x − xₗ)^β  (left singularity)',
                'right': 'w(x) = (xᵣ − x)^β  (right singularity)',
                'jacobi': 'w(x) = (x − xₗ)^α (xᵣ − x)^β  (Jacobi)'
            }[x],
            key="osc_weight_type"
        )
    
    alpha_w = 0.0
    beta_w = 0.0
    alpha_w_str = "0"
    beta_w_str = "0"
    
    with col_w2:
        if weight_type == 'jacobi':
            alpha_w_str = st.text_input(
                "α (left exponent)",
                value="-1/2",
                help="Symbolic expression for (x − xₗ)^α, must be > −1.  Examples: -1/2, -1/4, 0.3",
                key="osc_alpha"
            )
            try:
                alpha_w = float(sp.sympify(alpha_w_str))
            except Exception:
                st.error(f"Cannot parse α = `{alpha_w_str}`")
                alpha_w = -0.5
        
        if weight_type in ['left', 'right', 'jacobi']:
            beta_w_str = st.text_input(
                "β (exponent)",
                value="-1/2",
                help="Symbolic expression for weight exponent, must be > −1.  Examples: -1/4, -1/2, 1/3",
                key="osc_beta"
            )
            try:
                beta_w = float(sp.sympify(beta_w_str))
            except Exception:
                st.error(f"Cannot parse β = `{beta_w_str}`")
                beta_w = -0.5
    
    # Validate exponent range
    if weight_type in ['left', 'right'] and beta_w <= -1:
        st.error("β must be > −1 for integrability.")
    if weight_type == 'jacobi':
        if alpha_w <= -1:
            st.error("α must be > −1 for integrability.")
        if beta_w <= -1:
            st.error("β must be > −1 for integrability.")
    
    # Display weight function formula — show symbolic form in LaTeX
    try:
        _alpha_latex = sp.latex(sp.sympify(alpha_w_str)) if weight_type == 'jacobi' else "0"
    except Exception:
        _alpha_latex = str(alpha_w)
    try:
        _beta_latex = sp.latex(sp.sympify(beta_w_str)) if weight_type in ['left', 'right', 'jacobi'] else "0"
    except Exception:
        _beta_latex = str(beta_w)
    
    if weight_type == 'none':
        st.latex(r"w(x) = 1")
    elif weight_type == 'left':
        st.latex(rf"w(x) = (x - x_\ell)^{{{_beta_latex}}}")
    elif weight_type == 'right':
        st.latex(rf"w(x) = (x_r - x)^{{{_beta_latex}}}")
    else:
        st.latex(rf"w(x) = (x - x_\ell)^{{{_alpha_latex}}}\,(x_r - x)^{{{_beta_latex}}}")
    
    st.markdown("---")
    
    # ==========================================================================
    # REFERENCE VALUE OPTIONS
    # ==========================================================================
    st.markdown("### Reference Value")
    
    ref_method = st.radio(
        "Reference computation method",
        ['self_conv', 'mpmath'],
        format_func=lambda x: {
            'self_conv': 'Self-convergence (compare n to 8n)',
            'mpmath': 'High-precision quadrature (mpmath)'
        }[x],
        key="osc_ref_method",
        horizontal=True,
        help="Self-convergence is fastest. mpmath is slow but accurate for validation."
    )
    
    mpmath_dps = 50
    
    if ref_method == 'mpmath':
        mpmath_dps = st.slider(
            "Working precision (decimal places for internal computation)",
            min_value=30, max_value=100, value=50,
            help="Higher = more accurate but slower. 50 decimal places gives ~16 correct digits in final result.",
            key="osc_mpmath_dps"
        )
        st.caption(
            f"Working precision: {mpmath_dps} decimal places "
            f"→ approximately {int(mpmath_dps * 0.9)} accurate digits"
        )
        st.warning(
            "**Note:** High-precision quadrature can be slow, especially "
            "for large ω. A progress indicator will be shown."
        )
    else:
        st.info(
            r"**Self-convergence:** Error = $|I(n) - I(8n)| / |I(8n)|$ "
            "— fastest and reliable for FC validation."
        )
    
    st.markdown("---")
    
    # ==========================================================================
    # CONVERGENCE STUDY TYPE
    # ==========================================================================
    st.markdown("### Convergence Study")
    
    study_type = st.radio(
        "Study type",
        ['n_conv', 'omega_conv'],
        format_func=lambda x: {
            'n_conv': 'Convergence in n (fixed ω)',
            'omega_conv': 'Error vs ω (fixed n)'
        }[x],
        horizontal=True,
        key="osc_study_type"
    )
    
    if study_type == 'n_conv':
        # Single row: n_min, n_max, ω
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            n_min_exp = st.selectbox(
                r"$n_{\min}$", list(range(3, 10)), index=0,
                format_func=lambda x: f"{2**x}", key="osc_n_min_exp"
            )
        with col_b:
            n_max_exp = st.selectbox(
                r"$n_{\max}$", list(range(6, 15)), index=2,
                format_func=lambda x: f"{2**x}", key="osc_n_max_exp"
            )
        with col_c:
            omega_fixed = st.number_input(
                "ω",
                value=100.0,
                min_value=0.1,
                step=10.0,
                help="Fixed oscillation frequency",
                key="osc_omega_fixed"
            )
        
        n_min_osc = 2 ** n_min_exp
        n_max_osc = 2 ** n_max_exp
        
        ns_test = []
        n_curr = n_min_osc
        while n_curr <= n_max_osc:
            ns_test.append(n_curr)
            n_curr *= 2
        
        st.info(f"**Grid sizes:** {ns_test}  |  **ω = {omega_fixed}**")
    
    else:  # omega_conv
        # Row 1: ω_min, ω_max
        col_a, col_b = st.columns(2)
        with col_a:
            omega_min = st.number_input(
                r"$\omega_{\min}$", value=10.0, min_value=0.1, step=10.0,
                key="osc_omega_min"
            )
        with col_b:
            omega_max = st.number_input(
                r"$\omega_{\max}$", value=1000.0, min_value=1.0, step=100.0,
                key="osc_omega_max"
            )
        
        # Row 2: n, Δω
        col_c, col_d = st.columns(2)
        with col_c:
            n_fixed_exp = st.selectbox(
                "Grid size n", list(range(5, 12)), index=2,
                format_func=lambda x: f"{2**x}", key="osc_n_fixed_exp"
            )
            n_fixed = 2 ** n_fixed_exp
        with col_d:
            omega_step = st.number_input(
                r"Step size $\Delta\omega$",
                value=1.0,
                min_value=0.01,
                step=1.0,
                help="Uniform spacing for ω sampling",
                key="osc_omega_step"
            )
        
        omegas_test = np.arange(omega_min, omega_max + omega_step * 0.5, omega_step)
        if len(omegas_test) == 0 or omegas_test[-1] < omega_max:
            omegas_test = np.append(omegas_test, omega_max)
        
        n_omega = len(omegas_test)
        st.info(
            rf"**n = {n_fixed}**  |  "
            rf"$\omega \in [{omega_min:.1f},\, {omega_max:.1f}]$ "
            rf"with $\Delta\omega = {omega_step}$  ({n_omega} points)"
        )
    
    # Show moment cache stats
    cache = get_moment_cache()
    if cache.size() > 0:
        st.caption(f"Moment cache: {cache.size()} entries stored (persisted to `moment_cache.json`)")
    
    st.markdown("---")
    
    # ==========================================================================
    # RUN COMPUTATION
    # ==========================================================================
    
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        run_clicked = st.button("Compute Oscillatory Integral", type="primary",
                                use_container_width=True, key="osc_compute")
    with col_btn2:
        if st.button("Clear Cache", key="osc_clear_cache"):
            cache.clear()
            st.success("Cache cleared.")
    
    if run_clicked:
        
        # Capture cache size before computation for toast message
        st.session_state['_osc_cache_size_before'] = cache.size()
        
        # Get function and parameters from config
        func_str = config.get('func_str', 'sin(2*pi*x)')
        xl = params.get('xl', 0.0)
        xr = params.get('xr', 1.0)
        method = config.get('method', 'Hermite-GP')
        r = config.get('r', 4)
        p_ratio = config.get('p', 1)
        q_ratio = config.get('q', 1)
        shift = config.get('shift', 0.0)
        
        # Parse function
        try:
            x_sym = sp.Symbol('x')
            expr = sympify(func_str)
            func = lambdify(x_sym, expr, modules=['numpy'])
        except Exception as e:
            st.error(f"Error parsing function: {e}")
            return
        
        with st.spinner("Computing oscillatory integrals..."):
            
            if study_type == 'n_conv':
                # ==============================================================
                # N-CONVERGENCE STUDY
                # ==============================================================
                
                # Determine reference method
                if ref_method == 'mpmath':
                    ref_source = f"mpmath ({mpmath_dps} digits)"
                    use_self_conv = False
                    use_mpmath = True
                    ref_value = None
                else:
                    ref_source = "self-convergence (8n)"
                    use_self_conv = True
                    use_mpmath = False
                    ref_value = None
                
                results = {
                    'ns': [], 'values': [], 'errors': [], 'orders': [],
                    'reference': ref_value, 'ref_source': ref_source,
                    'omega': omega_fixed, 'references': []
                }
                
                # If mpmath, compute reference first
                if use_mpmath:
                    st.info("Computing high-precision reference integral with mpmath...")
                    mp_progress = st.progress(0)
                    try:
                        ref_value = compute_reference_integral_mpmath(
                            func_str, xl, xr, omega_fixed,
                            weight_type, alpha_w, beta_w, mpmath_dps
                        )
                        results['reference'] = ref_value
                        mp_progress.progress(100)
                        st.success(
                            f"Reference: {ref_value.real:.15e} + {ref_value.imag:.15e}i"
                        )
                    except Exception as e:
                        st.error(f"mpmath integration failed: {e}")
                        st.warning("Falling back to self-convergence.")
                        use_self_conv = True
                        use_mpmath = False
                        ref_source = "self-convergence (8n)"
                        results['ref_source'] = ref_source
                
                progress = st.progress(0)
                
                for i, n in enumerate(ns_test):
                    L = xr - xl
                    h = L / n
                    c = (p_ratio * n) // q_ratio
                    if (n + c) % 2 != 0:
                        c += 1
                    
                    use_n_plus_1 = abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12
                    if use_n_plus_1:
                        x_grid = xl + np.arange(n + 1) * h
                    else:
                        x_grid = xl + (np.arange(n) + shift) * h
                    
                    f_vals = np.array([func(x) for x in x_grid])
                    
                    if 'GP' in method.upper():
                        if 'Modulated' in method:
                            f_ext = app.extend_hermite_modulated(f_vals, c, r, shift, use_gp=True)
                        else:
                            f_ext = app.extend_hermite_gp(f_vals, c, r, shift)
                    else:
                        if 'Modulated' in method:
                            f_ext = app.extend_hermite_modulated(f_vals, c, r, shift, use_gp=False)
                        else:
                            f_ext = app.extend_hermite_proper(f_vals, c, r, shift)
                    
                    N = len(f_ext)
                    f_coeffs = np.fft.fft(f_ext) / N
                    
                    I = oscillatory_integral(
                        f_coeffs, xl, xr, omega_fixed, n, c,
                        weight_type, alpha_w, beta_w, shift
                    )
                    
                    results['ns'].append(n)
                    results['values'].append(I)
                    
                    if use_self_conv:
                        n_ref = 8 * n
                        h_ref = L / n_ref
                        c_ref = (p_ratio * n_ref) // q_ratio
                        if (n_ref + c_ref) % 2 != 0:
                            c_ref += 1
                        
                        if use_n_plus_1:
                            x_grid_ref = xl + np.arange(n_ref + 1) * h_ref
                        else:
                            x_grid_ref = xl + (np.arange(n_ref) + shift) * h_ref
                        
                        f_vals_ref = np.array([func(x) for x in x_grid_ref])
                        
                        if 'GP' in method.upper():
                            if 'Modulated' in method:
                                f_ext_ref = app.extend_hermite_modulated(
                                    f_vals_ref, c_ref, r, shift, use_gp=True)
                            else:
                                f_ext_ref = app.extend_hermite_gp(f_vals_ref, c_ref, r, shift)
                        else:
                            if 'Modulated' in method:
                                f_ext_ref = app.extend_hermite_modulated(
                                    f_vals_ref, c_ref, r, shift, use_gp=False)
                            else:
                                f_ext_ref = app.extend_hermite_proper(
                                    f_vals_ref, c_ref, r, shift)
                        
                        N_ref = len(f_ext_ref)
                        f_coeffs_ref = np.fft.fft(f_ext_ref) / N_ref
                        
                        I_ref = oscillatory_integral(
                            f_coeffs_ref, xl, xr, omega_fixed, n_ref, c_ref,
                            weight_type, alpha_w, beta_w, shift
                        )
                        
                        results['references'].append(I_ref)
                        err = (abs(I - I_ref) / abs(I_ref)
                               if abs(I_ref) > 1e-15 else abs(I - I_ref))
                    else:
                        results['references'].append(ref_value)
                        err = (abs(I - ref_value) / abs(ref_value)
                               if abs(ref_value) > 1e-15 else abs(I - ref_value))
                    
                    results['errors'].append(err)
                    progress.progress((i + 1) / len(ns_test))
                
                # Convergence orders
                for i in range(1, len(results['errors'])):
                    if results['errors'][i] > 0 and results['errors'][i-1] > 0:
                        order = np.log(results['errors'][i-1] / results['errors'][i]) / np.log(2)
                        results['orders'].append(order)
                    else:
                        results['orders'].append(np.nan)
                
                st.session_state.osc_results = results
                st.session_state.osc_results_study_type = 'n_conv'
                
            else:
                # ==============================================================
                # OMEGA SWEEP STUDY
                # ==============================================================
                n = n_fixed
                L = xr - xl
                h = L / n
                c = (p_ratio * n) // q_ratio
                if (n + c) % 2 != 0:
                    c += 1
                
                use_n_plus_1 = abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12
                if use_n_plus_1:
                    x_grid = xl + np.arange(n + 1) * h
                else:
                    x_grid = xl + (np.arange(n) + shift) * h
                
                f_vals = np.array([func(x) for x in x_grid])
                
                if 'GP' in method.upper():
                    if 'Modulated' in method:
                        f_ext = app.extend_hermite_modulated(f_vals, c, r, shift, use_gp=True)
                    else:
                        f_ext = app.extend_hermite_gp(f_vals, c, r, shift)
                else:
                    if 'Modulated' in method:
                        f_ext = app.extend_hermite_modulated(f_vals, c, r, shift, use_gp=False)
                    else:
                        f_ext = app.extend_hermite_proper(f_vals, c, r, shift)
                
                N = len(f_ext)
                f_coeffs = np.fft.fft(f_ext) / N
                
                # Determine reference source based on user's choice
                use_self_conv_omega = (ref_method == 'self_conv')
                use_mpmath_omega = (ref_method == 'mpmath')
                
                if use_mpmath_omega:
                    omega_ref_source = f"mpmath ({mpmath_dps} digits)"
                else:
                    omega_ref_source = "self-convergence (8n)"
                
                results = {
                    'n': n, 'omegas': [], 'values': [], 'errors': [],
                    'references': [], 'ref_source': omega_ref_source
                }
                
                # Precompute 8n reference grid if using self-convergence
                f_coeffs_ref = None
                n_ref = None
                c_ref = None
                if use_self_conv_omega:
                    n_ref = 8 * n
                    h_ref = L / n_ref
                    c_ref = (p_ratio * n_ref) // q_ratio
                    if (n_ref + c_ref) % 2 != 0:
                        c_ref += 1
                    
                    if use_n_plus_1:
                        x_grid_ref = xl + np.arange(n_ref + 1) * h_ref
                    else:
                        x_grid_ref = xl + (np.arange(n_ref) + shift) * h_ref
                    
                    f_vals_ref = np.array([func(x) for x in x_grid_ref])
                    
                    if 'GP' in method.upper():
                        if 'Modulated' in method:
                            f_ext_ref = app.extend_hermite_modulated(
                                f_vals_ref, c_ref, r, shift, use_gp=True)
                        else:
                            f_ext_ref = app.extend_hermite_gp(f_vals_ref, c_ref, r, shift)
                    else:
                        if 'Modulated' in method:
                            f_ext_ref = app.extend_hermite_modulated(
                                f_vals_ref, c_ref, r, shift, use_gp=False)
                        else:
                            f_ext_ref = app.extend_hermite_proper(
                                f_vals_ref, c_ref, r, shift)
                    
                    N_ref = len(f_ext_ref)
                    f_coeffs_ref = np.fft.fft(f_ext_ref) / N_ref
                
                progress = st.progress(0)
                
                for i, omega in enumerate(omegas_test):
                    # Compute FC approximation at n
                    I = oscillatory_integral(
                        f_coeffs, xl, xr, omega, n, c,
                        weight_type, alpha_w, beta_w, shift
                    )
                    
                    # Compute reference based on chosen method
                    if use_mpmath_omega:
                        try:
                            ref = compute_reference_integral_mpmath(
                                func_str, xl, xr, omega,
                                weight_type, alpha_w, beta_w, mpmath_dps
                            )
                        except Exception:
                            ref = I  # fallback: zero error
                    else:
                        ref = oscillatory_integral(
                            f_coeffs_ref, xl, xr, omega, n_ref, c_ref,
                            weight_type, alpha_w, beta_w, shift
                        )
                    
                    results['omegas'].append(omega)
                    results['values'].append(I)
                    results['references'].append(ref)
                    
                    err = abs(I - ref) / abs(ref) if abs(ref) > 1e-15 else abs(I - ref)
                    results['errors'].append(err)
                    
                    progress.progress((i + 1) / len(omegas_test))
                
                st.session_state.osc_results = results
                st.session_state.osc_results_study_type = 'omega_conv'
        
        # Save cache to disk and report
        old_size = st.session_state.get('_osc_cache_size_before', 0)
        new_size = cache.size()
        new_entries = new_size - old_size
        if cache.has_unsaved():
            saved = cache.save()
            if saved and new_entries > 0:
                st.toast(
                    f"{new_entries} new moment(s) saved to cache "
                    f"(total: {new_size} entries)"
                )
        st.success("Computation complete!")
    
    # ==========================================================================
    # DISPLAY RESULTS
    # ==========================================================================
    
    if 'osc_results' in st.session_state:
        results = st.session_state.osc_results
        
        st.markdown("---")
        st.markdown("### Results")
        
        if st.session_state.osc_results_study_type == 'n_conv':
            _display_n_convergence_results(results)
        else:
            _display_omega_results(results)
    
    # ==========================================================================
    # ABOUT SECTION
    # ==========================================================================
    st.markdown("---")
    
    with st.expander("**About Oscillatory Integration**", expanded=False):
        
        st.markdown("#### Method Overview")
        st.markdown(
            r"This tab computes high-frequency oscillatory integrals using "
            r"Fourier-based quadrature with FC extension."
        )
        st.latex(
            r"I = \int_{x_\ell}^{x_r} w(x)\, f(x)\, e^{i\omega x}\, dx "
            r"\;\approx\; \sum_{k} c_k \cdot W_{\omega,k}"
        )
        st.markdown(
            r"Here $c_k$ are the Fourier coefficients of $f$ obtained via FFT "
            r"of the FC-extended data, and $W_{\omega,k}$ are analytically computed **moments**."
        )
        
        st.markdown("#### Moment Formula")
        st.markdown(
            r"With grid shift $s = 0$ (grid points $x_j = x_\ell + j\,h$), "
            r"the FC basis function is $\phi_k(x) = e^{2\pi i k (x - x_\ell)/((n+c)\,h)}$ "
            r"and the moment is"
        )
        st.latex(
            r"W_{\omega,k} = \int_{x_\ell}^{x_r} w(x)\, e^{i\omega x}\,"
            r"e^{2\pi i k (x - x_\ell)/((n+c)\,h)}\, dx"
        )
        st.markdown(
            r"Substituting $u = (x - x_\ell)/(x_r - x_\ell)$ and writing "
            r"$L = x_r - x_\ell$, $h = L/n$, the combined phase exponent becomes "
            r"$i\,\gamma_k\, u$ where"
        )
        st.latex(
            r"\gamma_k = L\,\omega + \frac{2\pi k\, n}{n + c}"
        )
        st.markdown(
            r"With non-zero shift $s$ (grid points $x_j = x_\ell + (j+s)\,h$), "
            r"the basis picks up a phase factor:"
        )
        st.latex(
            r"W_{\omega,k}^{(s)} = e^{-2\pi i k s/(n+c)} \;\cdot\; W_{\omega,k}^{(0)}"
        )
        st.markdown(
            r"The base moment ($s = 0$) is cached; the shift correction is a cheap multiplication."
        )
        
        st.markdown("#### Weight Functions")
        st.markdown(
            r"The method supports Jacobi-type weights with exponents $> -1$. "
            r"Exponents can be entered as symbolic expressions (e.g. `-1/4`, `1/3`)."
        )
        st.markdown(
            r"""
| Type | Weight $w(x)$ | Use case |
|:-----|:--------------|:---------|
| None | $1$ | Standard oscillatory integral |
| Left | $(x - x_\ell)^\beta$ | Left endpoint singularity |
| Right | $(x_r - x)^\beta$ | Right endpoint singularity |
| Jacobi | $(x - x_\ell)^\alpha\,(x_r - x)^\beta$ | General Jacobi weight |
"""
        )
        st.markdown(
            r"For symmetric singularities $[(x - x_\ell)(x_r - x)]^\beta$, "
            r"select Jacobi with $\alpha = \beta$."
        )
        
        st.markdown("#### Moment Computation")
        st.markdown(
            r"Moments are evaluated analytically using `mpmath` high-precision arithmetic."
        )
        st.markdown(
            r"**Left singularity** $w(x) = (x - x_\ell)^\beta$: the key integral is"
        )
        st.latex(
            r"\int_0^1 u^\beta\, e^{i\gamma u}\, du "
            r"= \left(\frac{i}{\gamma}\right)^{\!\beta+1} "
            r"\gamma\!\left(\beta+1,\, -i\gamma\right)"
        )
        st.markdown(
            r"where $\gamma(a, z) = \Gamma(a) - \Gamma(a, z)$ is the lower incomplete gamma function."
        )
        st.markdown(
            r"**Right singularity** $w(x) = (x_r - x)^\beta$: "
            r"uses $\gamma(\beta+1,\, i\gamma)$ with an additional factor $e^{i\gamma_k}$."
        )
        st.markdown(
            r"**Jacobi weight** $w(x) = (x - x_\ell)^\alpha (x_r - x)^\beta$: "
            r"uses the confluent hypergeometric function"
        )
        st.latex(
            r"{}_1F_1\!\left(1 + \alpha,\; 2 + \alpha + \beta,\; i\gamma_k\right)"
        )
        
        st.markdown("#### Moment Cache")
        st.markdown(
            "Computed moments are stored in a JSON database (`moment_cache.json`) "
            "so that repeated runs with the same parameters reuse previously computed values. "
            "The cache persists across sessions. Use the **Clear Cache** button to reset it."
        )
        
        st.markdown("#### Convergence Studies")
        st.markdown(
            r"**$n$-convergence (fixed $\omega$):** "
            r"Refines the grid $n = 2^3, 2^4, \ldots$ at fixed frequency. "
            r"Shows spectral (exponential) convergence for smooth $f$."
        )
        st.markdown(
            r"**$\omega$-sweep (fixed $n$):** "
            r"Varies $\omega$ at fixed grid size. Reports the asymptotic convergence rate, "
            r"i.e. the slope $p$ in $\text{error} \sim \omega^{\,p}$, "
            r"fitted from the large-$\omega$ portion of the data."
        )


# ==========================================================================
# DISPLAY FUNCTIONS
# ==========================================================================

def _display_n_convergence_results(results):
    """Display results from n-convergence study."""
    
    ref_source = results.get('ref_source', 'self-convergence (8n)')
    if ref_source == 'self-convergence (8n)':
        st.info(r"Using **self-convergence**: error = $|I(n) - I(8n)| / |I(8n)|$")
        ref_str = "I(8n) per grid"
    elif 'mpmath' in ref_source:
        st.success(f"Using **high-precision** reference ({ref_source})")
        ref = results['reference']
        ref_str = f"{ref.real:.15e} + {ref.imag:.15e}i"
    else:
        st.info(f"Using reference: {ref_source}")
        ref = results.get('reference')
        ref_str = (f"{ref.real:.10e} + {ref.imag:.10e}i" if ref is not None else "N/A")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Reference", ref_str)
    
    with col2:
        valid_errors = [e for e in results['errors'] if np.isfinite(e) and e > 0]
        best_err = min(valid_errors) if valid_errors else float('nan')
        st.metric("Best Error", f"{best_err:.2e}" if np.isfinite(best_err) else "N/A")
    
    with col3:
        valid_orders = [o for o in results['orders'] if np.isfinite(o)]
        avg_order = np.mean(valid_orders) if valid_orders else 0
        st.metric("Avg Order", f"{avg_order:.2f}")
    
    # Table
    st.markdown("#### Convergence Table")
    
    if 'references' in results and results['references']:
        df = pd.DataFrame({
            'n': results['ns'],
            'Re(I)': [f"{v.real:.6e}" for v in results['values']],
            'Im(I)': [f"{v.imag:.6e}" for v in results['values']],
            'Re(Ref)': [f"{r.real:.6e}" for r in results['references']],
            'Im(Ref)': [f"{r.imag:.6e}" for r in results['references']],
            'Rel Error': [f"{e:.2e}" for e in results['errors']],
            'Order': (['—'] +
                      [f"{o:.2f}" if not np.isnan(o) else '—' for o in results['orders']])
        })
    else:
        df = pd.DataFrame({
            'n': results['ns'],
            'Re(I)': [f"{v.real:.6e}" for v in results['values']],
            'Im(I)': [f"{v.imag:.6e}" for v in results['values']],
            'Rel Error': [f"{e:.2e}" for e in results['errors']],
            'Order': (['—'] +
                      [f"{o:.2f}" if not np.isnan(o) else '—' for o in results['orders']])
        })
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Plots
    st.markdown("#### Convergence Plots")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    valid_mask = [np.isfinite(e) and e > 0 for e in results['errors']]
    valid_ns = [n for n, v in zip(results['ns'], valid_mask) if v]
    valid_errors = [e for e, v in zip(results['errors'], valid_mask) if v]
    
    if valid_errors:
        ax1.loglog(valid_ns, valid_errors, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Grid size n', fontsize=12)
    ax1.set_ylabel('Relative error', fontsize=12)
    ax1.set_title(f'Convergence (ω = {results["omega"]:.1f})', fontsize=14)
    ax1.grid(True, alpha=0.3, which='both')
    
    if results['orders']:
        valid_orders = [(n, o) for n, o in zip(results['ns'][1:], results['orders'])
                        if np.isfinite(o)]
        if valid_orders:
            n_mid, orders = zip(*valid_orders)
            ax2.plot(n_mid, orders, 'rs-', linewidth=2, markersize=8)
            mean_order = np.mean(orders)
            if np.isfinite(mean_order):
                ax2.axhline(mean_order, color='green', linestyle='--',
                           label=f'Average: {mean_order:.2f}')
                ax2.legend()
    ax2.set_xlabel('Grid size n', fontsize=12)
    ax2.set_ylabel('Convergence order', fontsize=12)
    ax2.set_title('Convergence Rate', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        create_download_button(fig, "oscillatory_convergence", key="dl_osc_conv")
    with col_dl2:
        # Build full-precision CSV for download
        df_dl = pd.DataFrame({
            'n': results['ns'],
            'Re(I)': [v.real for v in results['values']],
            'Im(I)': [v.imag for v in results['values']],
            'Re(Ref)': [r.real for r in results['references']] if results['references'] else [np.nan]*len(results['ns']),
            'Im(Ref)': [r.imag for r in results['references']] if results['references'] else [np.nan]*len(results['ns']),
            'Rel_Error': results['errors'],
            'Order': [np.nan] + [o for o in results['orders']]
        })
        csv_data = df_dl.to_csv(index=False)
        st.download_button(
            label="Download Data (CSV)",
            data=csv_data,
            file_name="oscillatory_n_convergence.csv",
            mime="text/csv",
            key="dl_osc_conv_csv",
            use_container_width=True
        )
    plt.close(fig)


def _display_omega_results(results):
    """Display results from omega sweep study."""
    
    n = results['n']
    ref_source = results.get('ref_source', 'self-convergence (8n)')
    
    if 'mpmath' in ref_source:
        st.success(f"Using **high-precision** reference ({ref_source})")
    else:
        st.info(f"**Reference:** FC approximation at n = {8*n} (8× finer grid)")
    
    omegas = np.array(results['omegas'])
    errors = np.array(results['errors'])
    
    # --- Item 5: compute asymptotic convergence rate ---
    valid_mask = np.array([np.isfinite(e) and e > 0 for e in errors])
    valid_omegas = omegas[valid_mask]
    valid_errors = errors[valid_mask]
    
    asymp_rate = np.nan
    asymp_rate_str = "N/A"
    if len(valid_omegas) >= 4:
        # Use the last half of data for asymptotic rate (large ω regime)
        n_half = len(valid_omegas) // 2
        log_om = np.log(valid_omegas[n_half:])
        log_err = np.log(valid_errors[n_half:])
        
        # Check that log_err is finite
        finite_mask = np.isfinite(log_err) & np.isfinite(log_om)
        if np.sum(finite_mask) >= 2:
            coeffs = np.polyfit(log_om[finite_mask], log_err[finite_mask], 1)
            asymp_rate = coeffs[0]
            asymp_rate_str = f"{asymp_rate:.2f}"
    
    # Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Grid size", f"n = {n}")
    
    with col2:
        if 'self-convergence' in ref_source:
            st.metric("Reference grid", f"n = {8*n}")
        else:
            st.metric("Reference", ref_source)
    
    with col3:
        max_err = max(valid_errors) if len(valid_errors) > 0 else float('nan')
        st.metric("Max error", f"{max_err:.2e}" if np.isfinite(max_err) else "N/A")
    
    with col4:
        st.metric("Asymptotic rate", asymp_rate_str,
                  help="Slope of log(error) vs log(ω) in the large-ω regime")
    
    # Plots
    st.markdown("#### Error vs Frequency")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error vs omega (log-log)
    if len(valid_omegas) > 0:
        ax1.loglog(valid_omegas, valid_errors, 'bo-', linewidth=2, markersize=4)
        
        # Plot asymptotic fit line
        if np.isfinite(asymp_rate) and len(valid_omegas) >= 4:
            n_half = len(valid_omegas) // 2
            om_fit = valid_omegas[n_half:]
            log_om = np.log(om_fit)
            log_err = np.log(valid_errors[n_half:])
            finite_mask = np.isfinite(log_err) & np.isfinite(log_om)
            if np.sum(finite_mask) >= 2:
                coeffs = np.polyfit(log_om[finite_mask], log_err[finite_mask], 1)
                err_fit = np.exp(np.polyval(coeffs, np.log(om_fit)))
                ax1.loglog(om_fit, err_fit, 'r--', linewidth=1.5,
                          label=rf'slope $\approx {asymp_rate:.2f}$')
                ax1.legend(fontsize=11)
    
    ax1.set_xlabel('Frequency ω', fontsize=12)
    ax1.set_ylabel('Relative error', fontsize=12)
    ax1.set_title(f'Error vs Frequency (n = {n})', fontsize=14)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Real and imaginary parts
    ax2.plot(omegas, [v.real for v in results['values']], 'b-',
             linewidth=2, label='Approx (Re)')
    ax2.plot(omegas, [r.real for r in results['references']], 'b--',
             linewidth=1, label='Ref (Re)')
    ax2.plot(omegas, [v.imag for v in results['values']], 'r-',
             linewidth=2, label='Approx (Im)')
    ax2.plot(omegas, [r.imag for r in results['references']], 'r--',
             linewidth=1, label='Ref (Im)')
    ax2.set_xlabel('Frequency ω', fontsize=12)
    ax2.set_ylabel('Integral value', fontsize=12)
    ax2.set_title('Integral Components', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        create_download_button(fig, "oscillatory_omega_sweep", key="dl_osc_omega")
    with col_dl2:
        df_dl = pd.DataFrame({
            'omega': results['omegas'],
            'Re(I)': [v.real for v in results['values']],
            'Im(I)': [v.imag for v in results['values']],
            'Re(Ref)': [r.real for r in results['references']],
            'Im(Ref)': [r.imag for r in results['references']],
            'Rel_Error': results['errors']
        })
        csv_data = df_dl.to_csv(index=False)
        st.download_button(
            label="Download Data (CSV)",
            data=csv_data,
            file_name="oscillatory_omega_sweep.csv",
            mime="text/csv",
            key="dl_osc_omega_csv",
            use_container_width=True
        )
    plt.close(fig)
    
    # --- Asymptotic rate detail ---
    if np.isfinite(asymp_rate):
        st.markdown(
            rf"**Asymptotic convergence rate:** error $\sim \omega^{{{asymp_rate:.2f}}}$ "
            rf"(slope of $\log(\mathrm{{error}})$ vs $\log(\omega)$ "
            f"fitted over the upper half of the frequency range)"
        )
