"""
HighFIE Lab - Compare Tab
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


def compare_tab():
    """Compare tab - add methods to compare against the Setup configuration."""
    
    st.markdown("### Compare Extension Methods")
    
    # Check if analysis has been run
    if not st.session_state.results_list or not st.session_state.analysis_params:
        st.warning("Please run analysis in the **Setup & Test** tab first!")
        st.markdown("""
        **To use comparison:**
        1. Go to **Setup & Test** tab
        2. Configure your function, domain, and grid settings
        3. Choose a baseline extension method
        4. Click **Run Complete Analysis**
        5. Come back here to add more methods for comparison
        """)
        return
    
    # Get baseline configuration from Setup
    params = st.session_state.analysis_params
    baseline_results = st.session_state.results_list
    
    st.success("Using configuration from Setup & Test tab")
    
    # Display baseline configuration (read-only)
    st.markdown("#### Baseline Configuration (from Setup & Test)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        func_display = params['func_str'][:50] + "..." if len(params['func_str']) > 50 else params['func_str']
        st.info(f"**Function**: `{func_display}`")
    
    with col2:
        st.info(f"**Domain**: [{params['xl']}, {params['xr']}]")
    
    with col3:
        st.info(f"**Grids**: {params['n_min']} to {params['n_max']}")
    
    # Show baseline method
    baseline_method = params['method']
    p = params['p']
    q = params['q']
    
    if baseline_method in ['Hermite', 'Hermite-FD']:
        baseline_label = f"Hermite-FD (r={params['r']}, c=({p}/{q})n)"
    elif baseline_method == 'Hermite-GP':
        d = params['r'] + 1  # Convert r back to d for display
        baseline_label = f"Hermite-GP (d={d}, c=({p}/{q})n)"
    elif baseline_method == 'Hermite-FD-Modulated':
        baseline_label = f"Hermite-FD-Modulated (r={params['r']}, c=({p}/{q})n)"
    elif baseline_method == 'Hermite-GP-Modulated':
        d = params['r'] + 1
        baseline_label = f"Hermite-GP-Modulated (d={d}, c=({p}/{q})n)"
    elif baseline_method == 'Custom':
        baseline_label = f"Custom Extension (c=({p}/{q})n)"
    elif baseline_method == 'Zero' and p == 0:
        baseline_label = "No Extension"
    else:
        baseline_label = f"{baseline_method} (c=({p}/{q})n)"
    
    st.info(f"**Baseline Method**: {baseline_label}")
    
    # Show modulation parameters if baseline is modulated
    if 'Modulated' in baseline_method and 'modulation_params' in params:
        r_base = params.get('r', 4)
        mod_left_vals = []
        mod_right_vals = []
        for m in range(r_base + 1):
            left_val = params['modulation_params'].get(f'mod_left_{m}', (m+1)/(r_base+2))
            right_val = params['modulation_params'].get(f'mod_right_{m}', (m+1)/(r_base+2))
            mod_left_vals.append(f"{left_val:.3f}")
            mod_right_vals.append(f"{right_val:.3f}")
        st.caption(f"**Modulation widths (Left)**: [{', '.join(mod_left_vals)}]")
        st.caption(f"**Modulation widths (Right)**: [{', '.join(mod_right_vals)}]")
    
    st.markdown("---")
    
    # Initialize comparison state
    if 'comparison_methods' not in st.session_state:
        st.session_state.comparison_methods = []
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {}
    if 'last_baseline_config' not in st.session_state:
        st.session_state.last_baseline_config = None
    
    # Create a hashable representation of current baseline config
    current_baseline_config = (
        baseline_label,
        params['func_str'],
        params['xl'],
        params['xr'],
        params['n_min'],
        params['n_max'],
        params['p'],
        params['q']
    )
    
    # Check if baseline configuration has changed - if so, reset comparison
    if st.session_state.last_baseline_config != current_baseline_config:
        st.session_state.comparison_results = {}
        st.session_state.comparison_methods = []
        st.session_state.last_baseline_config = current_baseline_config
        if st.session_state.last_baseline_config is not None:  # Don't show on first load
            st.info(f"Reset comparison for new configuration: {baseline_label}")
    
    # Add baseline to comparison results if not already there
    if baseline_label not in st.session_state.comparison_results:
        st.session_state.comparison_results[baseline_label] = {
            'config': params,
            'results': baseline_results
        }
    
    # ==========================================================================
    # METHOD SELECTION (REDESIGNED - SEPARATE CONTROLS)
    # ==========================================================================
    
    st.markdown("#### Method Selection")
    st.info(f"**Baseline**: {baseline_label} [Baseline] (always included)")
    
    st.markdown("**Configure method to add:**")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        # Method selection
        method_options = {
            "No Extension": "Zero",
            "Periodic": "Periodic",
            "Hermite-FD": "Hermite-FD",
            "Hermite-GP": "Hermite-GP",
            "Hermite-FD-Modulated": "Hermite-FD-Modulated",
            "Hermite-GP-Modulated": "Hermite-GP-Modulated"
        }
        
        selected_method_name = st.selectbox(
            "Extension Method",
            options=list(method_options.keys()),
            help="Select the extension method to compare"
        )
        
        selected_method = method_options[selected_method_name]
    
    with col2:
        # r parameter (only for Hermite methods)
        if selected_method in ["Hermite", "Hermite-FD", "Hermite-GP", "Hermite-FD-Modulated", "Hermite-GP-Modulated"]:
            max_r = 10 if "GP" in selected_method else 9
            label = "Degree (d)" if selected_method == "Hermite-GP" else "Order (r)"
            
            r_value = st.selectbox(
                label,
                options=list(range(2, max_r + 1)),
                index=2,  # Default to 4
                help="Hermite degree/order"
            )
            
            if selected_method == "Hermite-GP":
                r_value = r_value - 1  # Convert degree to order
        else:
            r_value = 0
            st.selectbox(
                "Order (r)",
                options=["N/A"],
                disabled=True,
                help="Only for Hermite methods"
            )
    
    with col3:
        # p parameter
        if selected_method_name == "No Extension":
            p_value = 0
            st.number_input("p", value=0, disabled=True, help="No extension (p=0)")
        else:
            p_value = st.number_input(
                "p",
                min_value=1,
                max_value=10,
                value=params['p'] if params['p'] > 0 else 1,
                help="Extension numerator: c = ⌊(p/q)×n⌋"
            )
    
    with col4:
        # q parameter  
        if selected_method_name == "No Extension":
            q_value = 1
            st.number_input("q", value=1, disabled=True, help="No extension (q=1)")
        else:
            q_value = st.number_input(
                "q",
                min_value=1,
                max_value=10,
                value=params['q'] if params['q'] > 0 else 1,
                help="Extension denominator: c = ⌊(p/q)×n⌋"
            )
    
    # Modulation parameters for modulated methods
    compare_mod_params = None
    if "Modulated" in selected_method:
        with st.expander("Modulation Parameters", expanded=False):
            st.caption("Set transition widths for each derivative order (0=sharp, 1=full extension)")
            
            # Initialize compare modulation params in session state if needed
            if 'compare_mod_params' not in st.session_state:
                st.session_state.compare_mod_params = {}
            
            compare_mod_params = {'left': [], 'right': []}
            
            # Determine actual r for modulated method
            actual_r = r_value
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Left**")
            with col_right:
                st.markdown("**Right**")
            
            for m in range(actual_r + 1):
                default_width = (m + 1) / (actual_r + 2)
                col_l, col_r = st.columns(2)
                
                with col_l:
                    key_l = f"compare_mod_left_{m}"
                    width_l = st.slider(
                        f"m={m}",
                        min_value=0.0,
                        max_value=1.0,
                        value=default_width,
                        step=0.025,
                        key=key_l,
                        label_visibility="collapsed" if m > 0 else "visible"
                    )
                    compare_mod_params['left'].append((0.0, width_l))
                
                with col_r:
                    key_r = f"compare_mod_right_{m}"
                    width_r = st.slider(
                        f"m={m} R",
                        min_value=0.0,
                        max_value=1.0,
                        value=default_width,
                        step=0.025,
                        key=key_r,
                        label_visibility="collapsed"
                    )
                    compare_mod_params['right'].append((0.0, width_r))
    
    # Create unique configuration key
    if selected_method_name == "No Extension":
        config_key = "No Extension"
    elif selected_method in ["Hermite", "Hermite-FD"]:
        config_key = f"Hermite-FD (r={r_value}, c=({p_value}/{q_value})n)"
    elif selected_method == "Hermite-GP":
        d_value = r_value + 1  # Convert back to degree for display
        config_key = f"Hermite-GP (d={d_value}, c=({p_value}/{q_value})n)"
    elif selected_method == "Hermite-FD-Modulated":
        # Include modulation info in key
        mod_summary = ",".join([f"{p[1]:.2f}" for p in compare_mod_params['left']]) if compare_mod_params else "default"
        config_key = f"Hermite-FD-Mod (r={r_value}, c=({p_value}/{q_value})n, w=[{mod_summary}])"
    elif selected_method == "Hermite-GP-Modulated":
        d_value = r_value + 1
        mod_summary = ",".join([f"{p[1]:.2f}" for p in compare_mod_params['left']]) if compare_mod_params else "default"
        config_key = f"Hermite-GP-Mod (d={d_value}, c=({p_value}/{q_value})n, w=[{mod_summary}])"
    else:
        config_key = f"{selected_method_name} (c=({p_value}/{q_value})n)"
    
    # Check if this configuration already exists
    config_exists = config_key in st.session_state.comparison_results
    
    col_add, col_info = st.columns([1, 2])
    
    with col_add:
        if config_exists:
            st.button(
                "Already Added",
                disabled=True,
                use_container_width=True,
                help="This exact configuration is already in the comparison"
            )
        else:
            add_config = st.button(
                "Add to Comparison",
                type="primary",
                use_container_width=True,
                help="Add this configuration to the comparison"
            )
    
    with col_info:
        if config_exists:
            st.caption("This configuration is already being compared")
        else:
            st.caption(f"Will add: **{config_key}**")
    
    # Add configuration if button clicked
    if not config_exists and 'add_config' in locals() and add_config:
        # Build configuration
        new_config = {
            'method': selected_method,
            'r': r_value,
            'p': p_value,
            'q': q_value,
            'label': config_key,
            'xl': params['xl'],
            'xr': params['xr'],
            'n_min': params['n_min'],
            'n_max': params['n_max'],
            'shift': params.get('shift', 0.0)
        }
        
        # Add modulation params if using modulated method - use compare_mod_params from UI
        if "Modulated" in selected_method and compare_mod_params is not None:
            # Convert compare_mod_params to the format expected by extend_grid_python
            mod_params_dict = {}
            for m, (_, width) in enumerate(compare_mod_params['left']):
                mod_params_dict[f'mod_left_{m}'] = width
            for m, (_, width) in enumerate(compare_mod_params['right']):
                mod_params_dict[f'mod_right_{m}'] = width
            new_config['modulation_params'] = mod_params_dict
        
        # Run computation for this configuration
        with st.spinner(f"Computing {config_key}..."):
            # Parse function from params
            func_str = params['func_str']
            
            # Check if it's Python code (contains 'def f(x):')
            if 'def f(x):' in func_str or 'def f(x)' in func_str:
                # Python code - execute it
                namespace = {'np': np, 'numpy': np}
                exec(func_str, namespace)
                if 'f' not in namespace:
                    st.error("Function code must define 'f'")
                    return
                func = namespace['f']
            else:
                # Expression - use sympify
                x_sym = sp.Symbol('x')
                expr = sympify(func_str)
                func = lambdify(x_sym, expr, modules=['numpy'])
            
            # Create app instance for extension computations
            app = FourierInterpolationApp()
            
            results_list = []
            n = params['n_min']
            
            while n <= params['n_max']:
                # Compute c using floor formula
                c = int((p_value / q_value) * n)
                
                # Compute grid with shift parameter
                h = (params['xr'] - params['xl']) / n
                shift = params.get('shift', 0.0)
                
                # Special case: s=0 or s=1 → use n+1 points (both endpoints)
                if abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12:
                    x_grid = params['xl'] + np.arange(n + 1) * h
                else:
                    x_grid = params['xl'] + (np.arange(n) + shift) * h
                
                f_vals = func(x_grid)
                
                # Extend and compute Fourier coefficients
                try:
                    shift_val = params.get('shift', 0.0)
                    # Use new_config for modulation_params (if present), combined with baseline params
                    config_for_extension = new_config if 'modulation_params' in new_config else params
                    extended, f_hat = app.compute_extension_and_fourier(
                        f_vals, params['xl'], params['xr'], n, c, selected_method, r_value, shift_val, config_for_extension
                    )
                except Exception as e:
                    st.error(f"Error computing {config_key}: {e}")
                    break
                
                # Evaluate on fine grid for convergence analysis
                n_fine = 2 * params['n_max']
                x_fine = np.linspace(params['xl'], params['xr'], n_fine)
                f_true = func(x_fine)
                
                period = (params['xr'] - params['xl']) * (1 + c / n)
                f_approx = app.fourier_eval_with_period(f_hat, x_fine, params['xl'], period, shift_val)
                
                # Compute errors
                abs_error = np.abs(f_approx - f_true)
                max_abs_error = np.max(abs_error)
                
                max_f_orig = np.max(np.abs(f_vals))
                max_rel_error = max_abs_error / max_f_orig if max_f_orig > 0 else 0
                
                # Extended domain error
                period_extended = (params['xr'] - params['xl']) * (1 + c / n)
                n_ext = n + c
                x_grid_ext = params['xl'] + (np.arange(n_ext) + shift) * h
                f_vals_ext = func(x_grid_ext)
                max_f_ext = np.max(np.abs(f_vals_ext))
                max_rel_error_ext = max_abs_error / max_f_ext if max_f_ext > 0 else 0
                
                results_list.append({
                    'n': n,
                    'c': c,
                    'h': h,
                    'x_grid': x_grid,
                    'f_vals': f_vals,
                    'extended': extended,
                    'coeffs': f_hat,
                    'max_abs_error': max_abs_error,
                    'max_rel_error': max_rel_error,
                    'max_rel_error_extended': max_rel_error_ext,
                    # Store detailed arrays for plotting
                    'x_fine': x_fine,
                    'f_true': f_true,
                    'f_approx': f_approx,
                    'abs_error': abs_error
                })
                
                n *= 2
            
            # Store results
            st.session_state.comparison_results[config_key] = {
                'config': new_config,
                'results': results_list
            }
            
            st.success(f"Added {config_key} to comparison!")
            st.rerun()
    
    # ==========================================================================
    # CURRENTLY COMPARING
    # ==========================================================================
    
    st.markdown("---")
    st.markdown("#### Currently Comparing")
    
    if len(st.session_state.comparison_results) == 1:
        st.info("Only baseline included. Add methods above to compare.")
    else:
        # Show current configurations with remove buttons
        configs_to_display = [(k, v) for k, v in st.session_state.comparison_results.items()]
        
        # Separate baseline and other methods
        baseline_config = None
        other_configs = []
        
        for label, data in configs_to_display:
            if label == baseline_label:
                baseline_config = (label, data)
            else:
                other_configs.append((label, data))
        
        # Row 1: Baseline only (full width or centered)
        if baseline_config:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"**[Baseline] {baseline_config[0]}**")
                # Show modulation params if baseline is modulated
                if 'modulation_params' in params and 'Modulated' in baseline_config[0]:
                    mod_vals = []
                    r_base = params.get('r', 4)
                    for m in range(r_base + 1):
                        left_val = params['modulation_params'].get(f'mod_left_{m}', (m+1)/(r_base+2))
                        mod_vals.append(f"{left_val:.2f}")
                    st.caption(f"Modulation widths: [{', '.join(mod_vals)}]")
                else:
                    st.caption("(Baseline from Setup & Test)")
        
        # Rows 2+: Other methods (3 per row) with fixed height containers
        if other_configs:
            st.markdown("**Added Methods:**")
            cols_per_row = 3
            for i in range(0, len(other_configs), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(other_configs):
                        label, data = other_configs[idx]
                        
                        with col:
                            # Use container for consistent alignment
                            with st.container():
                                # Truncate long labels for display
                                display_label = label if len(label) <= 40 else label[:37] + "..."
                                st.markdown(f"**{display_label}**")
                                if st.button(f"Remove", key=f"remove_{idx}", type="secondary", use_container_width=True):
                                    del st.session_state.comparison_results[label]
                                    st.rerun()
    
    st.markdown("---")
    
    # Clear all button
    if st.button("Clear All Comparisons", type="secondary", use_container_width=False, help="Remove all methods except baseline"):
        st.session_state.comparison_results = {
            baseline_label: {
                'config': params,
                'results': baseline_results
            }
        }
        st.rerun()
    
    # ==========================================================================
    # DISPLAY COMPARISON RESULTS
    # ==========================================================================
    
    if len(st.session_state.comparison_results) > 1:  # More than just baseline
        st.markdown("---")
        st.subheader("Comparison Results")
        
        st.info(f"Comparing {len(st.session_state.comparison_results)} method(s) including baseline")
        
        # Build comparison table
        comparison_data = []
        for label, data in st.session_state.comparison_results.items():
            results = data['results']
            finest = results[-1]
            config = data['config']
            
            # Compute average rate
            rates = []
            for i in range(len(results) - 1):
                if results[i]['max_rel_error'] > 0 and results[i+1]['max_rel_error'] > 0:
                    rate = np.log(results[i]['max_rel_error'] / results[i+1]['max_rel_error']) / np.log(2)
                    if np.isfinite(rate):
                        rates.append(rate)
            avg_rate = np.mean(rates) if rates else 0
            
            is_baseline = (label == baseline_label)
            
            comparison_data.append({
                'Method': label + (' [Baseline]' if is_baseline else ''),
                'Extension (c)': finest['c'],
                'Best Grid': finest['n'],
                'Max Abs Error': finest['max_abs_error'],
                'Max Rel Error': finest['max_rel_error'],
                'Avg Rate': avg_rate,
                'Quality': ('Excellent' if finest['max_rel_error'] < 1e-10 else
                           'Good' if finest['max_rel_error'] < 1e-6 else
                           'Fair' if finest['max_rel_error'] < 1e-3 else 'Poor')
            })
        
        # Create DataFrame and highlight winner
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        
        # Find winner (lowest error)
        winner_idx = df['Max Rel Error'].idxmin()
        
        # Display table with styling
        def highlight_rows(row):
            """Highlight winner and baseline rows."""
            # row.name is the index (row number)
            is_winner = (row.name == winner_idx)
            is_baseline_row = '[Baseline]' in str(row['Method'])
            
            if is_winner:
                return ['background-color: #d4edda'] * len(row)  # Green for winner
            elif is_baseline_row:
                return ['background-color: #fff3cd'] * len(row)  # Yellow for baseline
            else:
                return [''] * len(row)
        
        st.dataframe(
            df.style.apply(highlight_rows, axis=1).format({
                'Max Abs Error': '{:.2e}',
                'Max Rel Error': '{:.2e}',
                'Avg Rate': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown(f"**Winner**: {df.loc[winner_idx, 'Method']} | [Baseline] = Baseline from Setup")
        
        # Detailed visualizations
        st.markdown("---")
        st.markdown("#### Comparison Plots")
        
        # Plot 1: Convergence comparison
        fig_conv = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 2, figure=fig_conv, wspace=0.3)
        
        # Left: Convergence plot
        ax1 = fig_conv.add_subplot(gs[0, 0])
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(st.session_state.comparison_results)))
        
        for idx, (label, data) in enumerate(st.session_state.comparison_results.items()):
            results = data['results']
            grid_sizes = [r['n'] for r in results]
            rel_errors = [r['max_rel_error'] for r in results]
            
            # Highlight baseline
            linewidth = 3 if label == baseline_label else 2
            alpha = 1.0 if label == baseline_label else 0.7
            marker = 's' if label == baseline_label else 'o'
            
            # Create shorter label for legend (split into two lines if too long)
            short_label = label + (' (baseline)' if label == baseline_label else '')
            if len(short_label) > 30 and '(' in short_label:
                parts = short_label.split('(', 1)
                short_label = parts[0].strip() + '\n(' + parts[1]
            
            ax1.loglog(grid_sizes, rel_errors, marker=marker, linestyle='-', 
                      color=colors[idx], linewidth=linewidth, markersize=8, 
                      label=short_label, 
                      alpha=alpha)
        
        ax1.set_xlabel('Grid size (n)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Max relative error (wrt original)', fontsize=13, fontweight='bold')
        ax1.set_title('Convergence Comparison', fontsize=15, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Right: Rate comparison
        ax2 = fig_conv.add_subplot(gs[0, 1])
        
        labels_list = []
        labels_formatted = []  # Two-line formatted labels
        avg_rates = []
        colors_list = []
        
        for idx, (label, data) in enumerate(st.session_state.comparison_results.items()):
            results = data['results']
            rates = []
            for i in range(len(results) - 1):
                if results[i]['max_rel_error'] > 0 and results[i+1]['max_rel_error'] > 0:
                    rate = np.log(results[i]['max_rel_error'] / results[i+1]['max_rel_error']) / np.log(2)
                    if np.isfinite(rate):
                        rates.append(rate)
            
            # Add star to baseline
            star = ' [B]' if label == baseline_label else ''
            
            # Create short label for inside bar
            if '(' in label:
                method_part = label.split('(')[0].strip()
                short_label = f"{method_part}{star}"
            else:
                short_label = label + star
            
            labels_list.append(label)
            labels_formatted.append(short_label)
            avg_rates.append(np.mean(rates) if rates else 0)
            colors_list.append(colors[idx])
        
        bars = ax2.barh(range(len(labels_formatted)), avg_rates, color=colors_list, alpha=0.7, height=0.6)
        ax2.set_yticks(range(len(labels_formatted)))
        ax2.set_yticklabels([''] * len(labels_formatted))  # Hide y-axis labels, we'll put them inside bars
        ax2.set_xlabel('Average convergence rate', fontsize=13, fontweight='bold')
        ax2.set_title('Rate Comparison', fontsize=15, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add method labels inside the bars and rates to the right
        for i, (bar, rate, label_text) in enumerate(zip(bars, avg_rates, labels_formatted)):
            # Method label inside the bar (left-aligned)
            ax2.text(0.05, bar.get_y() + bar.get_height() / 2,
                    label_text, ha='left', va='center', fontsize=9, 
                    fontweight='bold', color='white')
            # Rate value to the right of the bar
            if rate > 0:
                ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                        f'{rate:.1f}', ha='left', va='center', fontsize=10, 
                        fontweight='bold', color=colors_list[i])
        
        # Highlight best rate
        best_rate_idx = np.argmax(avg_rates)
        bars[best_rate_idx].set_alpha(1.0)
        bars[best_rate_idx].set_edgecolor('gold')
        bars[best_rate_idx].set_linewidth(3)
        
        # Adjust x-axis to show rate labels
        max_rate = max(avg_rates) if avg_rates else 1
        ax2.set_xlim(0, max_rate * 1.15)
        
        st.pyplot(fig_conv)
        create_download_button(fig_conv, "convergence_comparison", key="dl_conv_comp")
        plt.close(fig_conv)
        
        # Plot 2: Error distribution comparison
        st.markdown("---")
        st.markdown("#### Error Distribution (Finest Grid)")
        
        # Get finest grid from first method
        first_label = list(st.session_state.comparison_results.keys())[0]
        x_fine = st.session_state.comparison_results[first_label]['results'][-1]['x_fine']
        
        fig_error = plt.figure(figsize=(18, 5))
        gs_error = GridSpec(1, 3, figure=fig_error, wspace=0.3)
        
        # Absolute error
        ax1 = fig_error.add_subplot(gs_error[0, 0])
        for idx, (label, data) in enumerate(st.session_state.comparison_results.items()):
            abs_err = data['results'][-1]['abs_error']
            linewidth = 3 if label == baseline_label else 2
            alpha = 1.0 if label == baseline_label else 0.6
            ax1.plot(x_fine, abs_err, '-', color=colors[idx], linewidth=linewidth, 
                    label=label, alpha=alpha)
        
        ax1.set_xlabel('x', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Absolute error', fontsize=11, fontweight='bold')
        ax1.set_title('Absolute Error Comparison', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(params['xl'], params['xr'])
        
        # Approximation comparison
        ax2 = fig_error.add_subplot(gs_error[0, 1])
        
        # Plot true function
        f_true = st.session_state.comparison_results[first_label]['results'][-1]['f_true']
        ax2.plot(x_fine, f_true, 'k-', linewidth=2.5, label='True function', alpha=0.8)
        
        # Plot approximations
        for idx, (label, data) in enumerate(st.session_state.comparison_results.items()):
            f_approx = data['results'][-1]['f_approx']
            linestyle = '-' if label == baseline_label else '--'
            linewidth = 2 if label == baseline_label else 1.5
            alpha = 0.8 if label == baseline_label else 0.5
            ax2.plot(x_fine, f_approx, linestyle, color=colors[idx], linewidth=linewidth, 
                    label=f'{label}', alpha=alpha)
        
        ax2.set_xlabel('x', fontsize=11, fontweight='bold')
        ax2.set_ylabel('f(x)', fontsize=11, fontweight='bold')
        ax2.set_title('Approximation Comparison', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(params['xl'], params['xr'])
        
        # Relative error (log scale)
        ax3 = fig_error.add_subplot(gs_error[0, 2])
        max_f = np.max(np.abs(f_true))
        
        for idx, (label, data) in enumerate(st.session_state.comparison_results.items()):
            abs_err = data['results'][-1]['abs_error']
            rel_err = abs_err / max_f if max_f > 0 else abs_err
            linewidth = 3 if label == baseline_label else 2
            alpha = 1.0 if label == baseline_label else 0.6
            ax3.semilogy(x_fine, rel_err, '-', color=colors[idx], linewidth=linewidth, 
                        label=label, alpha=alpha)
        
        ax3.set_xlabel('x', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Relative error', fontsize=11, fontweight='bold')
        ax3.set_title('Relative Error Comparison (log)', fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3, which='both')
        ax3.set_xlim(params['xl'], params['xr'])
        
        st.pyplot(fig_error)
        create_download_button(fig_error, "error_distribution_comparison", key="dl_error_comp")
        plt.close(fig_error)
        
        # Export comparison data
        st.markdown("---")
        st.markdown("#### Export Comparison Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as CSV
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download Comparison Table (CSV)",
                data=csv_data,
                file_name="fourier_comparison.csv",
                mime="text/csv"
            )
    
    else:
        st.info("Add methods above to see comparison results. The baseline from Setup & Test is already included.")
