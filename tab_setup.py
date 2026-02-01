"""
HighFIE Lab - Setup & Test Tab
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sympy as sp
from sympy import sympify, lambdify
import pandas as pd
import io

from grid_extender import FourierInterpolationApp, GridExtender


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


# =============================================================================
# TAB FUNCTIONS
# =============================================================================

def setup_tab(app):
    """Setup tab with full-width configuration."""
    st.markdown("## Configure Your Analysis")
    st.markdown("Define your function, domain, extension method, and analysis parameters with full-width editors.")
    
    # Initialize session state for configuration
    if 'config' not in st.session_state:
        st.session_state.config = {
            'func_str': "sin(2*pi*x) * exp(-0.5*x)",
            'xl': 0.0,
            'xr': 1.0,
            'xl_str': '0',
            'xr_str': '1',
            'method': 'Hermite',
            'r': 4,
            'p': 1,
            'q': 1,
            'n_min': 8,
            'n_max': 64
        }
    
    # ==========================================================================
    # HANDLE PENDING OPTIMIZATION/RESET (must happen BEFORE any widgets)
    # ==========================================================================
    if 'pending_optimization' in st.session_state:
        pending = st.session_state.pending_optimization
        if 'modulation_params' not in st.session_state.config:
            st.session_state.config['modulation_params'] = {}
        for m in range(pending['r'] + 1):
            opt_right = pending['params']['right'][m][1]
            opt_left = pending['params']['left'][m][1]
            # Update config
            st.session_state.config['modulation_params'][f"mod_right_{m}"] = opt_right
            st.session_state.config['modulation_params'][f"mod_left_{m}"] = opt_left
            # Directly set slider widget keys to new values
            st.session_state[f"slider_mod_right_{m}"] = opt_right
            st.session_state[f"slider_mod_left_{m}"] = opt_left
        st.session_state.optimal_mod_info = pending['info']
        del st.session_state.pending_optimization
    
    if 'pending_reset' in st.session_state:
        pending_r = st.session_state.pending_reset
        if 'modulation_params' not in st.session_state.config:
            st.session_state.config['modulation_params'] = {}
        for m in range(pending_r + 1):
            default_width = (m + 1) / (pending_r + 2)
            # Update config
            st.session_state.config['modulation_params'][f"mod_left_{m}"] = default_width
            st.session_state.config['modulation_params'][f"mod_right_{m}"] = default_width
            # Directly set slider widget keys to new values
            st.session_state[f"slider_mod_right_{m}"] = default_width
            st.session_state[f"slider_mod_left_{m}"] = default_width
        if 'optimal_mod_info' in st.session_state:
            del st.session_state.optimal_mod_info
        del st.session_state.pending_reset
    
    # ==========================================================================
    # SECTION 1: FUNCTION DEFINITION
    # ==========================================================================
    st.markdown("---")
    st.subheader("1. Test Function")
    
    col_func_input, col_func_preview = st.columns([1, 1])
    
    with col_func_input:
        # Preset functions
        presets = {
            "sin(2πx)·exp(-0.5x)": "sin(2*pi*x) * exp(-0.5*x)",
            "Runge function": "1 / (1 + 25*x**2)",
            "Exponential decay": "exp(-5*x)",
            "Polynomial x³-x": "x**3 - x",
            "High frequency": "sin(10*pi*x)",
            "Gaussian": "exp(-50*(x-0.5)**2)",
            "Custom": None
        }
        
        preset_choice = st.selectbox(
            "Choose preset or custom",
            list(presets.keys()),
            index=0,
            help="Select a preset function or choose 'Custom' to define your own"
        )
        
        # Function input mode
        if preset_choice == "Custom":
            # Track previous mode to detect changes
            if 'prev_func_mode' not in st.session_state:
                st.session_state.prev_func_mode = "Simple Expression"
            
            func_mode = st.radio(
                "Input mode",
                ["Simple Expression", "Python Code"],
                horizontal=True,
                key="func_input_mode_radio"
            )
            
            # Detect mode change and reset function string ONLY if not loading
            if func_mode != st.session_state.prev_func_mode and 'loaded_func_code' not in st.session_state:
                st.session_state.prev_func_mode = func_mode
                # Reset to default for the new mode
                if func_mode == "Simple Expression":
                    st.session_state.config['func_str'] = "sin(2*pi*x) * exp(-0.5*x)"
                else:  # Python Code
                    st.session_state.config['func_str'] = """def f(x):
    x = np.atleast_1d(x)
    return np.sin(2*np.pi*x) * np.exp(-0.5*x)"""
            
            # Update previous mode
            st.session_state.prev_func_mode = func_mode
            
            if func_mode == "Simple Expression":
                with st.expander("Help: Simple Expressions"):
                    st.markdown("""
                    Use SymPy expressions with variable `x`:
                    - `sin(2*pi*x)` - Sine function
                    - `exp(-x)` - Exponential
                    - `x**2 + 1` - Polynomial
                    - `1/(1+x**2)` - Rational function
                    """)
                
                func_str = st.text_input(
                    "Function expression",
                    value=st.session_state.config['func_str'],
                    help="Enter a mathematical expression using x"
                )
                
                try:
                    x_sym = sp.Symbol('x')
                    expr = sympify(func_str)
                    func = lambdify(x_sym, expr, modules=['numpy'])
                    st.session_state.config['func_str'] = func_str
                except Exception as e:
                    st.error(f"Invalid expression: {e}")
                    return None, None, None
            
            else:  # Python Code
                with st.expander("Help: Python Code"):
                    st.markdown("""
                    **Piecewise function:**
                    ```python
                    def f(x):
                        x = np.atleast_1d(x)
                        return np.where(x < 0, x**2, np.sin(x))
                    ```
                    
                    **Fourier series:**
                    ```python
                    def f(x):
                        result = 0
                        for n in range(1, 20):
                            result += np.sin(n*np.pi*x) / n**2
                        return result
                    ```
                    """)
                
                # Check if we should load code (from button or upload)
                if 'loaded_func_code' in st.session_state and st.session_state.loaded_func_code is not None:
                    default_func_code = st.session_state.loaded_func_code
                else:
                    default_func_code = """def f(x):
    x = np.atleast_1d(x)
    return np.sin(2*np.pi*x) * np.exp(-0.5*x)"""
                
                func_code = st.text_area(
                    "Python function definition",
                    value=default_func_code,
                    height=250,
                    help="Define function f(x) using numpy operations"
                )
                
                # Save/Load/Upload buttons inline
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    save_name = st.text_input("Name to save:", value="my_function", key="save_func_name", label_visibility="collapsed", placeholder="Name to save")
                
                with col2:
                    if st.button("Save", key="save_func_btn", use_container_width=True):
                        if func_code and save_name:
                            st.session_state.saved_code_snippets[save_name] = {
                                'code': func_code,
                                'type': 'function'
                            }
                            st.success(f"Saved '{save_name}'")
                
                with col3:
                    if st.session_state.saved_code_snippets:
                        func_snippets = {k: v for k, v in st.session_state.saved_code_snippets.items() if v['type'] == 'function'}
                        if func_snippets:
                            selected = st.selectbox("Load:", list(func_snippets.keys()), key="load_func_select", label_visibility="collapsed")
                            if st.button("Load", key="load_func_btn", use_container_width=True):
                                st.session_state.loaded_func_code = func_snippets[selected]['code']
                                st.rerun()
                
                with col4:
                    uploaded = st.file_uploader("Upload", type=['py', 'txt'], key="upload_func_file", label_visibility="collapsed")
                    if uploaded:
                        # Check if this is a new upload using hash
                        file_hash = hash(uploaded.read())
                        uploaded.seek(0)  # Reset file pointer
                        
                        if 'last_uploaded_func_hash' not in st.session_state:
                            st.session_state.last_uploaded_func_hash = None
                        
                        if file_hash != st.session_state.last_uploaded_func_hash:
                            # New file uploaded
                            code = uploaded.read().decode('utf-8')
                            name = uploaded.name.replace('.py', '').replace('.txt', '')
                            st.session_state.saved_code_snippets[name] = {'code': code, 'type': 'function'}
                            st.session_state.loaded_func_code = code
                            st.session_state.last_uploaded_func_hash = file_hash
                            st.success(f"Uploaded '{name}'!")
                            st.rerun()
                
                func_str = func_code
                st.session_state.config['func_str'] = func_str
                
                try:
                    namespace = {'np': np, 'numpy': np}
                    exec(func_code, namespace)
                    if 'f' not in namespace:
                        st.error("Must define a function named 'f'")
                        return None, None, None
                    func = namespace['f']
                except Exception as e:
                    st.error(f"Function error: {e}")
                    return None, None, None
        else:
            # Use preset
            func_str = presets[preset_choice]
            st.session_state.config['func_str'] = func_str
            x_sym = sp.Symbol('x')
            expr = sympify(func_str)
            func = lambdify(x_sym, expr, modules=['numpy'])
    
    # ==========================================================================
    # SECTION 2: DOMAIN
    # ==========================================================================
    st.markdown("---")
    st.subheader("2. Domain")
    
    col1, col2 = st.columns(2)
    with col1:
        xl_str = st.text_input(
            "Left boundary (xₗ)", 
            value=str(st.session_state.config.get('xl_str', '0')),
            help="Enter a number or symbolic expression (e.g., 0, -pi, 1/3, -1)"
        )
        # Evaluate symbolic expression
        try:
            xl = float(sp.sympify(xl_str))
            st.session_state.config['xl'] = xl
            st.session_state.config['xl_str'] = xl_str
            st.caption(f"Evaluates to: {xl:.6f}")
        except Exception as e:
            st.error(f"Invalid expression: {e}")
            return None, None, None
            
    with col2:
        xr_str = st.text_input(
            "Right boundary (xᵣ)", 
            value=str(st.session_state.config.get('xr_str', '1')),
            help="Enter a number or symbolic expression (e.g., 1, pi, 2/3, sqrt(2))"
        )
        # Evaluate symbolic expression
        try:
            xr = float(sp.sympify(xr_str))
            st.session_state.config['xr'] = xr
            st.session_state.config['xr_str'] = xr_str
            st.caption(f"Evaluates to: {xr:.6f}")
        except Exception as e:
            st.error(f"Invalid expression: {e}")
            return None, None, None
    
    if xl >= xr:
        st.error("Left boundary must be less than right boundary")
        return None, None, None
    
    # Show function preview
    with col_func_preview:
        st.markdown("**Function Preview**")
        try:
            x_preview = np.linspace(xl, xr, 200)
            f_preview = func(x_preview)
            
            fig_preview, ax_preview = plt.subplots(figsize=(6, 4.5))
            ax_preview.plot(x_preview, f_preview, 'b-', linewidth=2)
            ax_preview.grid(True, alpha=0.3)
            ax_preview.set_xlabel('x', fontsize=10)
            ax_preview.set_ylabel('f(x)', fontsize=10)
            
            # Format interval symbolically when possible
            def format_value(val):
                """Format value symbolically if it's a simple fraction or integer."""
                if val == int(val):
                    return str(int(val))
                # Check for common fractions
                for denom in [2, 3, 4, 5, 6, 8, 10]:
                    for numer in range(-10*denom, 10*denom + 1):
                        if abs(val - numer/denom) < 1e-10:
                            if numer == 0:
                                return "0"
                            elif denom == 1:
                                return str(numer)
                            elif abs(numer) == 1 and denom == 1:
                                return str(numer)
                            else:
                                return f"{numer}/{denom}"
                # Check for multiples of pi
                if abs(val / np.pi - round(val / np.pi)) < 1e-10:
                    mult = int(round(val / np.pi))
                    if mult == 0:
                        return "0"
                    elif mult == 1:
                        return "π"
                    elif mult == -1:
                        return "-π"
                    else:
                        return f"{mult}π"
                return f"{val:.2f}"
            
            xl_str = format_value(xl)
            xr_str = format_value(xr)
            ax_preview.set_title(f'Function on [{xl_str}, {xr_str}]', fontsize=11, fontweight='bold')
            
            st.pyplot(fig_preview)
            create_download_button(fig_preview, "function_preview", key="dl_func_preview")
            plt.close(fig_preview)
            
        except Exception as e:
            st.error(f"Preview error: {e}")
    
    # ==========================================================================
    # SECTION 3: EXTENSION METHOD
    # ==========================================================================
    st.markdown("---")
    st.subheader("3. Extension Method")
    
    col_ext_config, col_ext_preview = st.columns([1, 1])
    
    with col_ext_config:
        # Extension mode selection
        extension_mode = st.radio(
            "Extension Mode",
            ["No Extension", "Built-in Methods", "Custom Code"],
            index=1,
            help="No Extension: c=0. Built-in: Predefined extensions. Custom Code: Define your own.",
            key="extension_mode_radio"
        )
        
        custom_extension_func = None
        custom_extension_params = {}
        
        if extension_mode == "No Extension":
            method = "Zero"
            r = 2
            st.session_state.config['method'] = method
            st.session_state.config['r'] = r
            st.info("No extension will be applied (c = 0)")
            # Set p and q to 0 for no extension
            p = 0
            q = 0
        
        elif extension_mode == "Built-in Methods":
            method = st.selectbox(
                "Method",
                ["Periodic", "Hermite-FD", "Hermite-GP", 
                 "Hermite-FD-Modulated", "Hermite-GP-Modulated"],
                index=2 if app.gram_loaded else 1,
                help="Periodic: tile the function. Hermite: smooth polynomial extension. Modulated: Hermite with optimizable spatial support."
            )
            st.session_state.config['method'] = method
            
            # Hermite order
            r = 4
            if method in ["Hermite-FD", "Hermite-GP", "Hermite-FD-Modulated", "Hermite-GP-Modulated"]:
                max_r = 9 if "GP" in method else 8
                label = "Degree (d)" if "GP" in method else "Order (r)"
                
                if "GP" in method:
                    d = st.slider(label, min_value=2, max_value=10, value=5, step=1)
                    r = d - 1  # Convert degree to order
                    
                    # Warn if d might be too large
                    n_min = st.session_state.config.get('n_min', 16)
                    if d > n_min:
                        st.warning(f"d={d} requires n ≥ {d}. Current n_min={n_min} may be too small. Increase n_min or reduce d.")
                else:
                    r = st.slider(label, min_value=2, max_value=8, value=st.session_state.config['r'], step=1)
                
                st.session_state.config['r'] = r
                
                # Show controls for modulated methods
                if "Modulated" in method:
                    st.markdown("**Modulation Parameters**")
                    st.caption("Control the spatial support of each derivative term. "
                              "Each derivative order m has a transition width that determines "
                              "where the modulation function η(x) goes from 1 to 0.")
                    
                    # Initialize modulation params in session state if not exists
                    if 'modulation_params' not in st.session_state.config:
                        st.session_state.config['modulation_params'] = {}
                    
                    # Get current p/q for extension length estimation
                    p_est = st.session_state.config.get('p', 1)
                    q_est = st.session_state.config.get('q', 1)
                    c_fraction = p_est / q_est  # c/n ratio
                    
                    # Create expander for modulation controls
                    with st.expander("Customize Modulation Widths", expanded=False):
                        st.markdown("""
                        **Transition Width:** Controls how far the modulation extends.
                        - **0.0**: Sharp cutoff at boundary
                        - **0.5**: Transitions over half the extension region
                        - **1.0**: Transitions over full extension region
                        
                        **Default:** Higher derivatives get wider support: `(m+1)/(r+2) × c`
                        """)
                        
                        modulation_params = {'left': [], 'right': []}
                        
                        # Create columns for left and right modulation
                        col_mod_left, col_mod_right = st.columns(2)
                        
                        with col_mod_left:
                            st.markdown("**Left Boundary (at x_r)**")
                            st.caption("Modulation transitions from x_r → x_r + width")
                        
                        with col_mod_right:
                            st.markdown("**Right Boundary (at x_l)**")
                            st.caption("Modulation transitions from x_l - width → x_l")
                        
                        # Create sliders for each derivative order
                        for m in range(r + 1):
                            # Default width as fraction of extension: (m+1)/(r+2)
                            default_width = (m + 1) / (r + 2)
                            
                            col_l, col_r = st.columns(2)
                            
                            with col_l:
                                # Left modulation width
                                key_left = f"mod_left_{m}"
                                
                                # Initialize if not exists
                                if key_left not in st.session_state.config['modulation_params']:
                                    st.session_state.config['modulation_params'][key_left] = default_width
                                
                                current_val_left = float(st.session_state.config['modulation_params'][key_left])
                                
                                width_left = st.slider(
                                    f"m={m} (Left)",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=current_val_left,
                                    step=0.025,
                                    key=f"slider_{key_left}",
                                    help=f"Transition width for {m}-th derivative at left boundary"
                                )
                                st.session_state.config['modulation_params'][key_left] = width_left
                            
                            with col_r:
                                # Right modulation width
                                key_right = f"mod_right_{m}"
                                
                                # Initialize if not exists
                                if key_right not in st.session_state.config['modulation_params']:
                                    st.session_state.config['modulation_params'][key_right] = default_width
                                
                                current_val_right = float(st.session_state.config['modulation_params'][key_right])
                                
                                width_right = st.slider(
                                    f"m={m} (Right)",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=current_val_right,
                                    step=0.025,
                                    key=f"slider_{key_right}",
                                    help=f"Transition width for {m}-th derivative at right boundary"
                                )
                                st.session_state.config['modulation_params'][key_right] = width_right
                        
                        # Add buttons for reset and optimize
                        st.markdown("**Optimization**")
                        col_reset, col_h2, col_hr, col_max = st.columns(4)
                        
                        with col_reset:
                            if st.button("Reset", key="reset_modulation", use_container_width=True,
                                        help="Reset to default values (m+1)/(r+2)"):
                                # Store pending reset - will be applied before sliders are created on rerun
                                st.session_state.pending_reset = r
                                st.rerun()
                        
                        # Helper function to run optimization with caching
                        def run_optimization(norm_type, hr_order=None):
                            try:
                                # Use a reasonable n for optimization (cap at 256 for speed)
                                # The optimal parameters are relatively insensitive to n
                                n_max_config = st.session_state.config.get('n_max', 64)
                                n_opt = min(n_max_config, 256)  # Cap at 256 for fast optimization
                                
                                p_opt = st.session_state.config.get('p', 1)
                                q_opt = st.session_state.config.get('q', 1)
                                c_opt = int((p_opt / q_opt) * n_opt)
                                shift_opt = st.session_state.config.get('shift', 0.0)
                                use_gp = "GP" in method
                                
                                # Create cache key based on relevant parameters
                                func_str = st.session_state.config.get('func_str', '')
                                hr_key = f"_hr{hr_order}" if hr_order else ""
                                cache_key = f"{func_str}_{r}_{p_opt}_{q_opt}_{shift_opt}_{use_gp}_{norm_type}{hr_key}_{xl}_{xr}"
                                
                                # Initialize cache if not exists
                                if 'optimization_cache' not in st.session_state:
                                    st.session_state.optimization_cache = {}
                                
                                # Check cache
                                if cache_key in st.session_state.optimization_cache:
                                    cached = st.session_state.optimization_cache[cache_key]
                                    st.session_state.pending_optimization = {
                                        'params': cached['params'],
                                        'info': cached['info'],
                                        'r': r
                                    }
                                    st.toast("Using cached optimization results")
                                    st.rerun()
                                    return
                                
                                h_opt = (xr - xl) / n_opt
                                use_n_plus_1_opt = abs(shift_opt) < 1e-12 or abs(shift_opt - 1.0) < 1e-12
                                
                                if use_n_plus_1_opt:
                                    x_opt = xl + np.arange(n_opt + 1) * h_opt
                                else:
                                    x_opt = xl + (np.arange(n_opt) + shift_opt) * h_opt
                                
                                f_opt = func(x_opt)
                                
                                norm_label = f"H^{hr_order}" if hr_order else norm_type
                                with st.spinner(f"Optimizing ({norm_label} norm) with n={n_opt}..."):
                                    optimal_params, opt_info = app.optimize_modulation_params(
                                        f_opt, c_opt, r, shift_opt, use_gp, norm_type=norm_type, hr_order=hr_order
                                    )
                                
                                # Store in cache
                                st.session_state.optimization_cache[cache_key] = {
                                    'params': optimal_params,
                                    'info': opt_info
                                }
                                
                                # Store pending optimization - will be applied before sliders are created on rerun
                                st.session_state.pending_optimization = {
                                    'params': optimal_params,
                                    'info': opt_info,
                                    'r': r
                                }
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Optimization failed: {e}")
                        
                        with col_h2:
                            if st.button("H² norm", key="optimize_h2", 
                                        use_container_width=True,
                                        help="Minimize H² Sobolev norm"):
                                run_optimization('h2')
                        
                        with col_hr:
                            # H^r where r is the Hermite order
                            if st.button(f"H^{r} norm", key="optimize_hr", 
                                        use_container_width=True,
                                        help=f"Minimize H^{r} Sobolev norm (matches Hermite order)"):
                                run_optimization('hr', hr_order=r)
                        
                        with col_max:
                            if st.button("Max norm", key="optimize_max", 
                                        use_container_width=True,
                                        help="Minimize max|extension| (smallest amplitude)"):
                                run_optimization('max')
                        
                        # Show optimization results if available
                        if 'optimal_mod_info' in st.session_state:
                            info = st.session_state.optimal_mod_info
                            if info.get('success', False):
                                norm_name = info.get('norm_type', 'H²')
                                initial = info.get('norm_initial', 0)
                                optimal = info.get('norm_optimal', 0)
                                # Show factor of improvement instead of percentage
                                factor = initial / optimal if optimal > 0 else float('inf')
                                st.success(f"{norm_name}: {initial:.2e} → {optimal:.2e} ({factor:.1f}x improvement)")
                            else:
                                st.warning(f"Optimization: {info.get('message', 'Unknown status')}")
                    
                    # Note: Modulation plots are now shown in the preview column
                    st.caption("See modulation function profiles in the preview panel")
                else:
                    # Clear modulation params if not using modulated method
                    if 'modulation_params' in st.session_state.config:
                        del st.session_state.config['modulation_params']
            
            # Show warning only if Gram data not loaded
            if "GP" in method and not app.gram_loaded:
                st.warning("Hermite-GP data files not found. Method will fall back to Hermite-FD.")
        
        else:  # Custom Code
            st.markdown("**Define custom extension:**")
            
            # Show examples
            with st.expander("Extension Examples"):
                st.markdown("""
                **Polynomial Extrapolation:**
                ```python
                def extend_custom(f, c, xl, xr, n, **params):
                    h = (xr - xl) / n
                    order = params.get('order', 2)
                    num_pts = min(order + 1, n)
                    x_fit = np.arange(n - num_pts, n) * h + xl
                    coeffs = np.polyfit(x_fit, f[-num_pts:], order)
                    x_ext = (n + np.arange(c)) * h + xl
                    extension = np.polyval(coeffs, x_ext)
                    return np.concatenate([f, extension])
                ```
                
                **Exponential Decay:**
                ```python
                def extend_custom(f, c, xl, xr, n, **params):
                    h = (xr - xl) / n
                    decay_rate = params.get('decay_rate', 2.0)
                    t = np.arange(1, c + 1) * h
                    extension = f[-1] * np.exp(-decay_rate * t)
                    return np.concatenate([f, extension])
                ```
                """)
            
            default_extension_code = """def extend_custom(f, c, xl, xr, n, **params):
    \"\"\"Custom extension method.\"\"\"
    h = (xr - xl) / n
    
    # Example: Linear extrapolation
    slope = (f[-1] - f[-2]) / h
    extension = f[-1] + slope * np.arange(1, c + 1) * h
    
    return np.concatenate([f, extension])"""
            
            # Check if we should load code
            if 'loaded_ext_code' in st.session_state and st.session_state.loaded_ext_code is not None:
                extension_code_value = st.session_state.loaded_ext_code
            else:
                extension_code_value = default_extension_code
            
            extension_code = st.text_area(
                "Extension code",
                value=extension_code_value,
                height=300,
                help="Define extend_custom(f, c, xl, xr, n, **params)"
            )
            
            # Save/Load/Upload buttons inline
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                save_ext_name = st.text_input("Name to save:", value="my_extension", key="save_ext_name", label_visibility="collapsed", placeholder="Name to save")
            
            with col2:
                if st.button("Save", key="save_ext_btn", use_container_width=True):
                    if extension_code and save_ext_name:
                        st.session_state.saved_code_snippets[save_ext_name] = {
                            'code': extension_code,
                            'type': 'extension'
                        }
                        st.success(f"Saved '{save_ext_name}'")
            
            with col3:
                if st.session_state.saved_code_snippets:
                    ext_snippets = {k: v for k, v in st.session_state.saved_code_snippets.items() if v['type'] == 'extension'}
                    if ext_snippets:
                        selected_ext = st.selectbox("Load:", list(ext_snippets.keys()), key="load_ext_select", label_visibility="collapsed")
                        if st.button("Load", key="load_ext_btn", use_container_width=True):
                            st.session_state.loaded_ext_code = ext_snippets[selected_ext]['code']
                            st.rerun()
            
            with col4:
                uploaded_ext = st.file_uploader("Upload", type=['py', 'txt'], key="upload_ext_file", label_visibility="collapsed")
                if uploaded_ext:
                    # Check if this is a new upload using hash
                    ext_file_hash = hash(uploaded_ext.read())
                    uploaded_ext.seek(0)  # Reset file pointer
                    
                    if 'last_uploaded_ext_hash' not in st.session_state:
                        st.session_state.last_uploaded_ext_hash = None
                    
                    if ext_file_hash != st.session_state.last_uploaded_ext_hash:
                        # New file uploaded
                        ext_code = uploaded_ext.read().decode('utf-8')
                        ext_name = uploaded_ext.name.replace('.py', '').replace('.txt', '')
                        st.session_state.saved_code_snippets[ext_name] = {'code': ext_code, 'type': 'extension'}
                        st.session_state.loaded_ext_code = ext_code
                        st.session_state.last_uploaded_ext_hash = ext_file_hash
                        st.success(f"Uploaded '{ext_name}'!")
                        st.rerun()
            
            # Optional parameters
            st.markdown("**Extension Parameters:**")
            num_params = st.number_input("Number of parameters", min_value=0, max_value=5, value=0, step=1)
            
            for i in range(num_params):
                col1, col2 = st.columns([2, 1])
                with col1:
                    param_name = st.text_input(f"Param {i+1} name", value=f"param{i+1}", key=f"pname_{i}")
                with col2:
                    param_value = st.number_input(f"Value", value=1.0, key=f"pval_{i}", format="%.3f")
                custom_extension_params[param_name] = param_value
            
            # Parse and validate
            try:
                namespace = {'np': np, 'numpy': np}
                exec(extension_code, namespace)
                
                if 'extend_custom' not in namespace:
                    st.error("Code must define 'extend_custom'")
                    return None, None, None
                
                custom_extension_func = namespace['extend_custom']
                app.set_custom_extension(custom_extension_func, custom_extension_params)
                method = "Custom"
                r = 4
                st.session_state.config['method'] = method
                
            except Exception as e:
                st.error(f"Code error: {e}")
                return None, None, None
    
    with col_ext_preview:
        st.markdown("**Extension Preview**")
        
        # Automatically show preview
        try:
            # Use n_max for preview (shows the grid we ultimately care about)
            n_test = st.session_state.config.get('n_max', 64)
            if extension_mode == "No Extension":
                c_test = 0
            else:
                # Get p and q from config (they'll be set in Section 4)
                p_preview = st.session_state.config.get('p', 1)
                q_preview = st.session_state.config.get('q', 1)
                c_test = int((p_preview / q_preview) * n_test)
            
            # Get shift for grid generation
            preview_shift_val = st.session_state.config.get('shift', 0.0)
            h_test_preview = (xr - xl) / n_test
            
            # Generate grid - use n+1 points for s=0 or s=1 (MATLAB-style)
            use_n_plus_1_preview = abs(preview_shift_val) < 1e-12 or abs(preview_shift_val - 1.0) < 1e-12
            
            if use_n_plus_1_preview:
                # n+1 points including both endpoints
                x_test = xl + np.arange(n_test + 1) * h_test_preview
            else:
                # n points with shift
                x_test = xl + (np.arange(n_test) + preview_shift_val) * h_test_preview
            
            f_test = func(x_test)
            
            # Run extension
            if extension_mode == "No Extension":
                # No extension, just use the original data
                extended_test = f_test
                st.caption(f"No extension (c = 0)")
            elif extension_mode == "Built-in Methods":
                preview_shift = st.session_state.config.get('shift', 0.0)
                # Pass config explicitly for modulation parameters
                extended_test = app.extend_grid_python(f_test, xl, xr, c_test, method, r, preview_shift, st.session_state.config)
                # Clean caption showing key parameters
                if "Modulated" in method:
                    st.caption(f"n={n_test} (n_max), c={c_test}, r={r} | {method}")
                else:
                    st.caption(f"n={n_test} (n_max), c={c_test} | {method}")
            else:
                if custom_extension_func is None:
                    st.info("Define custom extension code above to see preview")
                    return None, None, None
                extended_test = custom_extension_func(f_test, c_test, xl, xr, n_test, **custom_extension_params)
                st.caption(f"Preview with n={n_test}, c={c_test} (using p={p_preview}, q={q_preview})")
            
            # Validate
            # Note: For s=0 or s=1, f_test has n+1 points
            # Hermite methods trim to n points before extending → return n+c
            # Simple methods (Zero, Constant, etc.) don't trim → return (n+1)+c
            preview_s = st.session_state.config.get('shift', 0.0)
            use_n_plus_1 = abs(preview_s) < 1e-12 or abs(preview_s - 1.0) < 1e-12
            
            # Hermite methods handle n+1 correctly, others don't
            hermite_methods = ["Hermite", "Hermite-FD", "Hermite-GP", 
                              "Hermite-FD-Modulated", "Hermite-GP-Modulated"]
            method_trims = method in hermite_methods
            
            if use_n_plus_1 and not method_trims:
                # Simple methods don't trim: expect (n+1) + c
                expected_len = (n_test + 1) + c_test
            else:
                # Hermite methods or non-n+1 grids: expect n + c
                expected_len = n_test + c_test
            
            if not isinstance(extended_test, np.ndarray):
                st.error("Must return numpy array")
            elif len(extended_test) != expected_len:
                st.error(f"Wrong length: expected {expected_len}, got {len(extended_test)}")
            elif c_test > 0 and not np.allclose(extended_test[:n_test], f_test[:n_test], rtol=1e-10):
                st.error("First n elements must equal input (first n)")
            elif not np.all(np.isfinite(extended_test)):
                st.error("Contains NaN or Inf")
            else:
                # Bilateral preview - simplified and corrected
                fig_test, ax_test = plt.subplots(figsize=(7, 4))
                h_test = (xr - xl) / n_test
                
                # Get shift from config
                preview_s = st.session_state.config.get('shift', 0.0)
                
                # Determine grid type and method behavior
                use_n_plus_1 = abs(preview_s) < 1e-12 or abs(preview_s - 1.0) < 1e-12
                hermite_methods = ["Hermite", "Hermite-FD", "Hermite-GP", 
                                  "Hermite-FD-Modulated", "Hermite-GP-Modulated"]
                method_trims = method in hermite_methods
                
                # Determine n_orig: the number of "original" points in extended_test
                if use_n_plus_1 and not method_trims:
                    n_orig = n_test + 1  # Simple methods keep all n+1 points
                else:
                    n_orig = n_test  # Hermite methods trim to n points
                
                # x_test may have n+1 points; use first n_orig for plotting
                x_grid_plot = x_test[:n_orig]
                
                if c_test == 0:
                    # No extension - just plot original grid
                    ax_test.plot(x_grid_plot, extended_test[:len(x_grid_plot)], 'bo', markersize=6,
                               label='Input grid function', zorder=5)
                    ax_test.axvline(xl, color='gray', linestyle='--', alpha=0.6, linewidth=2,
                                  label='Domain boundaries')
                    ax_test.axvline(xr, color='gray', linestyle='--', alpha=0.6, linewidth=2)
                else:
                    # With extension - extended_test has n_orig + c_test points
                    # Extended grid points (right side)
                    x_right_ext = xl + (np.arange(n_orig, n_orig + c_test) + preview_s) * h_test
                    
                    # Extended grid points (left side - bilateral)
                    x_left_ext = xl - (np.arange(c_test, 0, -1) - preview_s) * h_test
                    
                    # Full extended grid for green curve
                    x_full = np.concatenate([x_left_ext, x_grid_plot, x_right_ext])
                    f_full = np.concatenate([extended_test[n_orig:], extended_test[:n_orig], extended_test[n_orig:]])
                    
                    # Green curve for extended function
                    ax_test.plot(x_full, f_full, 'g-', linewidth=1.5, alpha=0.7, 
                               label='Extended function', zorder=3)
                    
                    # Blue circles for input grid function
                    ax_test.plot(x_grid_plot, extended_test[:n_orig], 'bo', markersize=6, 
                               label='Input grid function', zorder=5)
                    
                    # Red squares for extended grid function  
                    ax_test.plot(x_left_ext, extended_test[n_orig:n_orig+c_test], 'rs', markersize=6, 
                               label='Extended grid function', zorder=5)
                    ax_test.plot(x_right_ext, extended_test[n_orig:n_orig+c_test], 'rs', markersize=6, zorder=5)
                    
                    # Extension regions
                    ax_test.axvspan(x_left_ext[0], xl, alpha=0.2, color='yellow', label='Extension region')
                    ax_test.axvspan(xr, x_right_ext[-1], alpha=0.2, color='yellow')
                    
                    # Domain boundaries
                    ax_test.axvline(xl, color='gray', linestyle='--', alpha=0.6, linewidth=2, 
                                  label='Domain boundaries')
                    ax_test.axvline(xr, color='gray', linestyle='--', alpha=0.6, linewidth=2)
                
                ax_test.legend(fontsize=8, ncol=2, loc='best')
                ax_test.grid(True, alpha=0.3)
                
                # Update title - always shows n (even though we might use n+1 for derivatives)
                ax_test.set_title(f'Extension Preview: n={n_test}, c={c_test}', 
                                fontsize=10, fontweight='bold')
                
                ax_test.set_xlabel('x', fontsize=9)
                ax_test.set_ylabel('f(x)', fontsize=9)
                st.pyplot(fig_test)
                create_download_button(fig_test, "extension_preview", key="dl_ext_preview")
                plt.close(fig_test)
                
                # ============================================================
                # MODULATION FUNCTION PLOTS (for modulated methods)
                # ============================================================
                if "Modulated" in method and 'modulation_params' in st.session_state.config:
                    st.markdown("---")
                    st.markdown("**Modulation Function Profiles**")
                    st.caption("Shows η(y) vs y ∈ [0,1] for each derivative order")
                    
                    fig_mod, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7, 3))
                    
                    y_plot = np.linspace(0, 1, 200)
                    
                    # Plot left boundary modulations
                    ax_left.set_title("Left Boundary η(y)", fontsize=9)
                    ax_left.set_xlabel("y", fontsize=8)
                    ax_left.set_ylabel("η(y)", fontsize=8)
                    ax_left.grid(True, alpha=0.3)
                    ax_left.set_ylim([-0.05, 1.05])
                    
                    for m in range(r + 1):
                        width_left = st.session_state.config['modulation_params'].get(f"mod_left_{m}", (m+1)/(r+2))
                        # Compute modulation function
                        eta_values = np.array([app.modulation_function(y, 0.0, width_left, r) for y in y_plot])
                        ax_left.plot(y_plot, eta_values, label=f"m={m}", linewidth=1.5)
                    
                    ax_left.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
                    ax_left.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
                    ax_left.legend(loc='best', fontsize=7)
                    ax_left.tick_params(labelsize=7)
                    
                    # Plot right boundary modulations
                    ax_right.set_title("Right Boundary η(y)", fontsize=9)
                    ax_right.set_xlabel("y", fontsize=8)
                    ax_right.set_ylabel("η(y)", fontsize=8)
                    ax_right.grid(True, alpha=0.3)
                    ax_right.set_ylim([-0.05, 1.05])
                    
                    for m in range(r + 1):
                        width_right = st.session_state.config['modulation_params'].get(f"mod_right_{m}", (m+1)/(r+2))
                        # Compute modulation function
                        eta_values = np.array([app.modulation_function(y, 0.0, width_right, r) for y in y_plot])
                        ax_right.plot(y_plot, eta_values, label=f"m={m}", linewidth=1.5)
                    
                    ax_right.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
                    ax_right.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
                    ax_right.legend(loc='best', fontsize=7)
                    ax_right.tick_params(labelsize=7)
                    
                    plt.tight_layout()
                    st.pyplot(fig_mod)
                    create_download_button(fig_mod, "modulation_profiles", key="dl_mod_preview")
                    plt.close(fig_mod)
        
        except Exception as e:
            st.error(f"Preview failed: {e}")
    
    # ==========================================================================
    # SECTION 4: GRID CONFIGURATION
    # ==========================================================================
    st.markdown("---")
    st.subheader("4. Grid Configuration")
    
    # Initialize shift parameter if not exists
    if 'shift' not in st.session_state.config:
        st.session_state.config['shift'] = 0.0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if extension_mode == "No Extension":
            # Force p=0 for no extension and disable input
            p = 0
            st.number_input("Extension numerator (p)", min_value=0, max_value=0, 
                          value=0, step=1, disabled=True, 
                          help="Disabled: No extension mode uses p=0")
            st.session_state.config['p'] = 0
        else:
            # Ensure valid default when switching from No Extension
            default_p = max(1, st.session_state.config['p'])
            p = st.number_input("Extension numerator (p)", min_value=1, max_value=32, 
                              value=default_p, step=1,
                              help="Extension size: c = (p/q) × n")
            st.session_state.config['p'] = p
    
    with col2:
        if extension_mode == "No Extension":
            # Force q=1 for no extension and disable input
            q = 1
            st.number_input("Extension denominator (q)", min_value=1, max_value=1, 
                          value=1, step=1, disabled=True,
                          help="Disabled: No extension mode uses q=1")
            st.session_state.config['q'] = 1
        else:
            # Ensure valid default when switching from No Extension
            default_q = max(1, st.session_state.config['q'])
            q = st.number_input("Extension denominator (q)", min_value=1, max_value=32, 
                              value=default_q, step=1,
                              help="Extension size: c = floor((p/q) × n)")
            st.session_state.config['q'] = q
    
    with col3:
        n_min = st.number_input("n min", min_value=4, max_value=128, 
                              value=st.session_state.config['n_min'], step=4,
                              help="Minimum grid size")
        st.session_state.config['n_min'] = n_min
    
    with col4:
        n_max = st.number_input("n max", min_value=8, max_value=16384, 
                              value=st.session_state.config['n_max'], step=8,
                              help="Maximum grid size (up to 2048)")
        st.session_state.config['n_max'] = n_max
    
    with col5:
        # Initialize shift_str if not exists
        if 'shift_str' not in st.session_state.config:
            st.session_state.config['shift_str'] = "0"
        
        shift_str = st.text_input("Grid shift (s ∈ [0,1])", 
                                 value=st.session_state.config['shift_str'],
                                 help="Grid points: xⱼ = x_ℓ + (j+s)h. Examples: 0, 1/2, 1/4, 0.25")
        
        # Parse symbolic input to float
        try:
            # Try to evaluate as fraction or expression
            if '/' in shift_str:
                parts = shift_str.split('/')
                if len(parts) == 2:
                    shift = float(parts[0]) / float(parts[1])
                else:
                    shift = float(eval(shift_str))
            else:
                shift = float(shift_str)
            
            # Validate range
            if shift < 0 or shift > 1:
                st.error("Shift must be between 0 and 1")
                shift = max(0, min(1, shift))
            
            st.session_state.config['shift'] = shift
            st.session_state.config['shift_str'] = shift_str  # Store the string representation
        except Exception as e:
            st.error(f"Invalid shift value: {shift_str}. Using default 0.")
            shift = 0.0
            st.session_state.config['shift'] = 0.0
            st.session_state.config['shift_str'] = "0"
    
    if n_min >= n_max:
        st.error("n_min must be < n_max")
        return None, None, None
    
    n_levels = int(np.log2(n_max / n_min)) + 1
    
    # Display grid info with shift parameter
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"Extension: c = ({p}/{q}) × n")
    with col2:
        grid_sizes = [n_min * 2**i for i in range(n_levels)]
        st.info(f"Grids: {', '.join(map(str, grid_sizes[:5]))}{'...' if n_levels > 5 else ''}")
    with col3:
        # Check if using n+1 grid (MATLAB-style)
        use_n_plus_1 = abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12
        
        if shift == 0:
            grid_type = "Standard (n+1 pts, both endpoints)" if use_n_plus_1 else "Standard"
        elif abs(shift - 1.0) < 1e-12:
            grid_type = "Shifted (n+1 pts, both endpoints)"
        elif shift == 0.5:
            grid_type = "Open (midpoints, n pts)"
        else:
            grid_type = f"Shifted (s={shift}, n pts)"
        st.info(f"Grid: {grid_type}")
    
    # ==========================================================================
    # RUN BUTTON
    # ==========================================================================
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_analysis = st.button(
            "Run Complete Analysis",
            type="primary",
            use_container_width=True,
            help="Execute analysis with current configuration",
            key="run_analysis_btn"
        )
    
    if run_analysis:
        with st.spinner("Running convergence analysis..."):
            results_list = []
            
            for level in range(n_levels):
                n_level = n_min * (2 ** level)
                
                # Use floor formula: c = floor(p/q * n)
                c_level = int((p / q) * n_level)
                
                # Generate grid with shift parameter
                # Special case: s=0 or s=1 → use n+1 points (both endpoints) like MATLAB
                if abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12:
                    # Grid with both endpoints: n+1 points over n intervals
                    h = (xr - xl) / n_level
                    x_grid = xl + np.arange(n_level + 1) * h
                else:
                    # Standard: n points without endpoint
                    h = (xr - xl) / n_level
                    x_grid = xl + (np.arange(n_level) + shift) * h
                
                f_vals = func(x_grid)
                
                # Extend and compute Fourier
                # Pass config explicitly for modulation parameters
                extended, coeffs = app.compute_extension_and_fourier(
                    f_vals, xl, xr, n_level, c_level, method, r, shift, st.session_state.config
                )
                
                extended_period = (xr - xl) * (1 + c_level / n_level)
                
                # Evaluate on fine grid for convergence analysis
                # n_eval = 2 * n_max (evaluation grid size)
                n_fine = 2 * n_max
                x_fine = np.linspace(xl, xr, n_fine)
                f_true = func(x_fine)
                f_approx = app.fourier_eval_with_period(coeffs, x_fine, xl, extended_period, shift)
                
                # Errors
                abs_error = np.abs(f_true - f_approx)
                max_abs_error = np.max(abs_error)
                
                max_f_orig = np.max(np.abs(f_vals))
                max_f_extended = np.max(np.abs(extended))
                
                max_rel_error = max_abs_error / max_f_orig if max_f_orig > 1e-15 else 0.0
                max_rel_error_extended = max_abs_error / max_f_extended if max_f_extended > 1e-15 else 0.0
                
                results_list.append({
                    'n': n_level,
                    'c': c_level,
                    'h': h,
                    'x_grid': x_grid,
                    'f_vals': f_vals,
                    'extended': extended,
                    'coeffs': coeffs,
                    'x_fine': x_fine,
                    'f_true': f_true,
                    'f_approx': f_approx,
                    'abs_error': abs_error,
                    'max_abs_error': max_abs_error,
                    'max_rel_error': max_rel_error,
                    'max_rel_error_extended': max_rel_error_extended,
                    'max_f_orig': max_f_orig,
                    'max_f_extended': max_f_extended,
                    'extended_period': extended_period
                })
            
            # Store results in BOTH for compatibility
            st.session_state.results = results_list
            st.session_state.results_list = results_list  # For Quick Test section
            st.session_state.analysis_params = {
                'xl': xl,
                'xr': xr,
                'func_str': func_str,
                'method': method,
                'extension_mode': extension_mode,
                'custom_extension_params': custom_extension_params if method == "Custom" else {},
                'r': r,
                'p': p,
                'q': q,
                'shift': shift,  # Add shift parameter
                'n_min': n_min,
                'n_max': n_max,
                'n_levels': n_levels
            }
            
            # Store modulation params if using modulated method
            if "Modulated" in method and 'modulation_params' in st.session_state.config:
                st.session_state.analysis_params['modulation_params'] = st.session_state.config['modulation_params'].copy()
            
            if not results_list:
                st.error("No valid grid sizes found! Adjust p and q.")
            else:
                st.success(f"Analysis complete! {len(results_list)} grids tested. Scroll down to see results.")
    
    return func, func_str, method


