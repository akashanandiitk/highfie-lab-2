"""
Fourier Interpolation with Grid Extension - Phase 3
Interactive Web Interface with Full-Width Configuration

Phase 3 Features:
- Setup tab with full-width code editors
- Two-column layouts (Config | Preview)
- Minimal sidebar
- Clean, modular structure
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sympy as sp
from sympy import sympify, lambdify
import pandas as pd
import math
import io

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="HighFIE Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* ========================================================================
       LIGHT MODE (Default) - Off-white background with clean styling
       ======================================================================== */
    
    /* Clean off-white background - matching academic webpage style */
    .stApp {
        background-color: #f9f9f9;
        background-image: none;
    }
    
    /* Make tabs more prominent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #e8e8e8;
        padding: 8px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 6px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e3a5f !important;
        color: white !important;
    }
    
    /* ========================================================================
       DARK MODE Support - Adapt colors for dark mode
       ======================================================================== */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #1a1a2e !important;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(26,26,46,0.8) 0%, transparent 50%),
                radial-gradient(circle at 90% 80%, rgba(26,26,46,0.8) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(20,20,40,0.4) 0%, transparent 70%),
                repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(30,30,50,0.1) 2px, rgba(30,30,50,0.1) 4px),
                repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(30,30,50,0.1) 2px, rgba(30,30,50,0.1) 4px);
        }
        
        /* Adjust text colors for dark mode */
        .main-header {
            color: #4da6ff !important;
        }
        
        /* Adjust info boxes for dark mode */
        .info-box {
            background-color: #16213e !important;
            border-left-color: #4da6ff !important;
        }
        
        .success-box {
            background-color: #1a3a1a !important;
            border-left-color: #4caf50 !important;
        }
        
        .warning-box {
            background-color: #3a2a1a !important;
            border-left-color: #ff9800 !important;
        }
        
        .error-box {
            background-color: #3a1a1a !important;
            border-left-color: #f44336 !important;
        }
        
        /* Input labels in dark mode */
        .stTextInput > label,
        .stNumberInput > label,
        .stSelectbox > label,
        .stMultiSelect > label,
        .stTextArea > label {
            color: #b0b0b0 !important;
        }
    }
    
    /* Completely hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Hide Streamlit footer and menu */
    footer {
        visibility: hidden;
    }
    #MainMenu {
        visibility: hidden;
    }
    header {
        visibility: hidden;
    }
    
    /* Expand main content to full width */
    .main .block-container {
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 1rem;
        background-color: transparent;
    }
    
    /* Dark theme for all input widgets (works in both light and dark mode) */
    /* Text inputs */
    input[type="text"],
    input[type="number"],
    textarea,
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: #e0e0e0 !important;
        border: 2px solid #0f3460 !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        font-weight: 500 !important;
        caret-color: #4da6ff !important;  /* Bright blue cursor for visibility */
    }
    
    /* Input focus state */
    input[type="text"]:focus,
    input[type="number"]:focus,
    textarea:focus,
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {
        border-color: #1f77b4 !important;
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2) !important;
    }
    
    /* Select boxes and multiselect */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: #e0e0e0 !important;
        border: 2px solid #0f3460 !important;
        border-radius: 8px !important;
    }
    
    /* Dropdown menus */
    [data-baseweb="popover"] {
        background: #16213e !important;
    }
    
    [role="option"] {
        background: #16213e !important;
        color: #e0e0e0 !important;
    }
    
    [role="option"]:hover {
        background: #1f77b4 !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: #e0e0e0 !important;
        border: 2px solid #0f3460 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        margin: 0.25rem !important;
    }
    
    /* Checkboxes */
    .stCheckbox > label {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: #e0e0e0 !important;
        border: 2px solid #0f3460 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Checkbox text and labels - ensure visibility */
    .stCheckbox label span,
    .stCheckbox label p,
    .stCheckbox div[data-testid="stMarkdownContainer"] p {
        color: #e0e0e0 !important;
    }
    
    /* Checkbox input elements */
    .stCheckbox input[type="checkbox"] {
        accent-color: #1f77b4 !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
    }
    
    /* Code editor */
    .stCodeBlock {
        background: #1a1a2e !important;
        border: 2px solid #0f3460 !important;
        border-radius: 8px !important;
    }
    
    /* Ensure cursor is visible in all text areas and code editors */
    * {
        caret-color: #4da6ff !important;  /* Bright blue cursor globally */
    }
    
    /* Override for light backgrounds (if any) */
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
        caret-color: #4da6ff !important;
    }
    
    /* Input labels */
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label,
    .stMultiSelect > label,
    .stTextArea > label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
    }
    
    /* Dark theme for comparison buttons */
    .stButton button[kind="secondary"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: #e0e0e0 !important;
        border: 2px solid #0f3460 !important;
    }
    
    .stButton button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%) !important;
        border-color: #1f77b4 !important;
    }
</style>
""", unsafe_allow_html=True)

# Import modules
from grid_extender import FourierInterpolationApp, GridExtender
from tab_setup import setup_tab
from tab_compare import compare_tab
from tab_oscillatory import oscillatory_integration_tab

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_download_button(fig, filename, label="Download Plot (PNG, 300 DPI)", key=None):
    """Create a download button for a matplotlib figure."""
    import io
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
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    
    # Initialize session state
    if 'results_list' not in st.session_state:
        st.session_state.results_list = []
    if 'analysis_params' not in st.session_state:
        st.session_state.analysis_params = {}
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {}
    
    # Create app instance
    app = FourierInterpolationApp()
    
    # Header
    st.markdown('<p class="main-header">HighFIE Lab</p>', unsafe_allow_html=True)
    st.markdown("**High**-Order **F**ourier **I**nterpolation with **E**xtension")
    
    # Custom styling for tabs
    st.markdown("""
    <style>
        div[data-testid="stHorizontalBlock"] > div:first-child {
            padding-right: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    with st.expander("**About the Method**", expanded=False):
        st.markdown('<div style="color: #4da6ff; font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem;">Method Overview</div>', unsafe_allow_html=True)
        
        st.markdown(r"""
        Given a grid function $f$ sampled on a uniform grid of size $n$ in the interval $[x_\ell, x_r]$, 
        this tool implements Fourier interpolation with grid extension using the following approach:
        """)
        
        st.markdown("#### **Grid Structure**")
        st.markdown("The uniform grid is defined as:")
        st.latex(r"x_j = x_\ell + (j + s)h, \quad j = 0, 1, \ldots, n-1")
        st.latex(r"h = \frac{x_r - x_\ell}{n}")
        
        st.markdown(r"""
        where $s$ is a shift parameter $(0 \le s \le 1)$:
        - $s = 0$: Standard closed grid (includes $x_\ell$)
        - $s = \frac{1}{2}$: Open grid (grid points at interval midpoints)
        - $s \in (0,1)$: Shifted grid (customizable positioning)
        """)
        
        st.markdown("#### **Extension Process**")
        st.markdown(r"""
        1. **Data Extension**: The method first extends the grid function by adding $c$ additional points 
           to create an extended grid of size $n + c$
        2. **Extension Size**: The extension parameter $c$ is typically chosen as a rational multiple of $n$:
        """)
        st.latex(r"c = \left\lfloor \frac{p}{q} \times n \right\rfloor")
        st.markdown(r"""
           where $q$ must divide $n$ for optimal FFT efficiency
        3. **Extension Methods**: Various approaches can be used:
           - Zero padding
           - Constant extension
           - Periodic extension
           - Linear extrapolation
           - Hermite interpolation (order $r$)
           - Custom user-defined extensions
        """)
        
        st.markdown("#### **Fourier Interpolation**")
        st.markdown(r"""
        Based on the extended data, the method constructs a Fourier interpolant as an approximation 
        to the input grid function using the Fast Fourier Transform (FFT).
        """)
        
        st.markdown("#### **Key Parameters**")
        st.markdown(r"""
        - $n$: Original grid size
        - $c$: Extension size $= \lfloor (p/q) \times n \rfloor$
        - $p, q$: Rational parameters controlling extension ratio
        - $s$: Grid shift parameter $(0 \le s \le 1)$
        - $r$: Hermite interpolation order (for Hermite method)
        - $[x_\ell, x_r]$: Computational domain interval
        """)
    
    # ==========================================================================
    # TABS - Now with 3 tabs including Oscillatory Integration
    # ==========================================================================
    
    tab_setup, tab_compare, tab_oscillatory = st.tabs([
        "Setup & Test",
        "Compare",
        "Oscillatory Integration"
    ])
    
    # TAB 1: SETUP & TEST
    with tab_setup:
        setup_tab(app)
        
        # Add quick test section at bottom
        st.markdown("---")
        st.markdown("### Quick Test")
        st.info("**Tip**: Test your configuration with a single method before running full comparison.")
        
        if st.session_state.results_list:
            st.markdown("#### Latest Test Results")
            
            # Quick summary metrics
            results = st.session_state.results_list
            finest = results[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Best Grid", f"n = {finest['n']}")
            
            with col2:
                st.metric("Max Abs Error", f"{finest['max_abs_error']:.2e}")
            
            with col3:
                st.metric("Max Rel Error", f"{finest['max_rel_error']:.2e}")
            
            with col4:
                # Compute average rate
                rates = []
                for i in range(len(results) - 1):
                    if results[i]['max_rel_error'] > 0 and results[i+1]['max_rel_error'] > 0:
                        rate = np.log(results[i]['max_rel_error'] / results[i+1]['max_rel_error']) / np.log(2)
                        if np.isfinite(rate):
                            rates.append(rate)
                avg_rate = np.mean(rates) if rates else 0
                st.metric("Avg Rate", f"{avg_rate:.2f}")
            
            # Grid selector
            st.markdown("#### Select Grid Size for Detailed Plots")
            selected_idx = st.selectbox(
                "Grid size:",
                range(len(results)),
                format_func=lambda i: f"n = {results[i]['n']}",
                index=len(results) - 1,
                help="Choose which grid size to visualize in detail",
                key="quick_test_grid_selector"
            )
            
            selected_result = results[selected_idx]
            
            # Show detailed plots
            st.markdown(f"#### Detailed Analysis (n = {selected_result['n']})")
            
            # Get parameters for plotting
            params = st.session_state.analysis_params
            xl = params['xl']
            xr = params['xr']
            
            # Create 3 rows of plots
            # Row 1: Extended grid and approximation (2 plots)
            fig1 = plt.figure(figsize=(18, 5))
            gs1 = GridSpec(1, 2, figure=fig1, wspace=0.25)
            
            # Plot 1: Extended grid function
            ax1 = fig1.add_subplot(gs1[0, 0])
            
            h = selected_result['h']
            n = selected_result['n']
            c = selected_result['c']
            x_grid = selected_result['x_grid']
            extended = selected_result['extended']
            s = st.session_state.config.get('shift', 0.0)
            
            # For s=0 or s=1, we use n+1 input points but only n points for FFT
            use_n_plus_1 = abs(s) < 1e-12 or abs(s - 1.0) < 1e-12
            
            # Determine the actual number of input function values used
            n_input = n + 1 if use_n_plus_1 else n
            
            # x_grid has n_input points, extended has n+c points
            # For plotting the input grid, use x_grid
            x_grid_plot = x_grid[:n_input]
            
            # The extended array structure: first n points are the trimmed input, last c are extension
            # But extended always has exactly n+c points
            
            if c == 0:
                ax1.plot(x_grid_plot[:n], extended, 'bo', markersize=6,
                        label='Input grid function', zorder=5)
                ax1.axvline(xl, color='gray', linestyle='--', alpha=0.6, linewidth=2,
                           label='Domain boundaries')
                ax1.axvline(xr, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            else:
                # Extension region coordinates
                # For s=0 or s=1: shift is effectively 0 for extension points
                s_eff = s if not use_n_plus_1 else 0.0
                
                x_right_ext = xl + (np.arange(n, n + c) + s_eff) * h
                x_left_ext = xl - (np.arange(c, 0, -1) - s_eff) * h
                
                # Original grid points (first n of the extended array)
                x_orig = xl + (np.arange(n) + s_eff) * h
                
                # Build full extended grid for plotting the green curve
                x_full = np.concatenate([x_left_ext, x_orig, x_right_ext])
                # extended[:n] are original, extended[n:] are extension
                f_full = np.concatenate([extended[n:], extended[:n], extended[n:]])
                
                # Ensure arrays match
                if len(x_full) == len(f_full):
                    ax1.plot(x_full, f_full, 'g-', linewidth=1.5, alpha=0.7, 
                             label='Extended function', zorder=3)
                
                # Plot input grid function (blue circles)
                ax1.plot(x_orig, extended[:n], 'bo', markersize=6, 
                         label='Input grid function', zorder=5)
                
                # Plot extension points (red squares)
                ax1.plot(x_left_ext, extended[n:n+c], 'rs', markersize=6, 
                         label='Extended grid function', zorder=5)
                ax1.plot(x_right_ext, extended[n:n+c], 'rs', markersize=6, zorder=5)
                
                # Extension regions
                ax1.axvspan(x_left_ext[0], xl, alpha=0.2, color='yellow', 
                           label='Extension region')
                ax1.axvspan(xr, x_right_ext[-1], alpha=0.2, color='yellow')
                ax1.axvline(xl, color='gray', linestyle='--', alpha=0.6, linewidth=2, 
                           label='Domain boundaries')
                ax1.axvline(xr, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            
            ax1.set_xlabel('x', fontsize=11, fontweight='bold')
            ax1.set_ylabel('f(x)', fontsize=11, fontweight='bold')
            ax1.set_title(f'Extended Grid Function (n={n}, c={c})', fontsize=13, fontweight='bold')
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Function vs Approximation
            ax2 = fig1.add_subplot(gs1[0, 1])
            
            ax2.plot(selected_result['x_fine'], selected_result['f_true'], 'b-', linewidth=2, 
                    label='True function', alpha=0.7)
            ax2.plot(selected_result['x_fine'], selected_result['f_approx'], 'r--', linewidth=2, 
                    label='Fourier approximation', alpha=0.7)
            # Use x_grid_plot which has the correct size matching f_vals
            f_vals = selected_result['f_vals']
            x_for_plot = x_grid[:len(f_vals)]  # Ensure matching sizes
            ax2.plot(x_for_plot, f_vals, 'go', markersize=5, label='Input grid function', zorder=5)
            
            ax2.set_xlabel('x', fontsize=11, fontweight='bold')
            ax2.set_ylabel('f(x)', fontsize=11, fontweight='bold')
            ax2.set_title('True Function vs Fourier Approximation', fontsize=13, fontweight='bold')
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(xl, xr)
            
            plt.tight_layout()
            st.pyplot(fig1)
            create_download_button(fig1, f"extended_grid_approximation_n{selected_result['n']}", key="dl_fig1")
            plt.close(fig1)
            
            # Row 2: Error profiles (2 plots)
            fig2 = plt.figure(figsize=(18, 5))
            gs2 = GridSpec(1, 2, figure=fig2, wspace=0.25)
            
            # Plot 3: Absolute error
            ax3 = fig2.add_subplot(gs2[0, 0])
            ax3.plot(selected_result['x_fine'], selected_result['abs_error'], 'r-', linewidth=2, alpha=0.7)
            ax3.axhline(selected_result['max_abs_error'], color='k', linestyle='--', 
                       label=f"Max = {selected_result['max_abs_error']:.2e}")
            ax3.set_xlabel('x', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Absolute error', fontsize=11, fontweight='bold')
            ax3.set_title('Absolute Error: |f(x) - f̃(x)|', fontsize=13, fontweight='bold')
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(xl, xr)
            
            # Plot 4: Relative errors
            ax4 = fig2.add_subplot(gs2[0, 1])
            
            max_f_orig = selected_result['max_f_orig']
            pointwise_rel = selected_result['abs_error'] / max_f_orig
            ax4.semilogy(selected_result['x_fine'], pointwise_rel, 'r-', linewidth=2, 
                        label=f'Rel error: ÷ max|f(x_i)| on orig grid = {max_f_orig:.3f}')
            
            max_f_extended = selected_result['max_f_extended']
            pointwise_rel_ext = selected_result['abs_error'] / max_f_extended
            ax4.semilogy(selected_result['x_fine'], pointwise_rel_ext, 'g-', linewidth=2, 
                        alpha=0.7, label=f'Rel error (ext): ÷ max|extended(x_i)| = {max_f_extended:.3f}')
            
            ax4.axhline(y=selected_result['max_rel_error'], color='r', linestyle='--', alpha=0.5, linewidth=1,
                       label=f'Max rel error = {selected_result["max_rel_error"]:.2e}')
            ax4.axhline(y=selected_result['max_rel_error_extended'], color='g', linestyle='--', alpha=0.5, linewidth=1,
                       label=f'Max rel error (ext) = {selected_result["max_rel_error_extended"]:.2e}')
            
            ax4.set_xlabel('x', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Relative Error', fontsize=11, fontweight='bold')
            ax4.set_title(f'Relative Approximation Errors (n={selected_result["n"]})', fontsize=13, fontweight='bold')
            ax4.legend(loc='best', fontsize=9)
            ax4.grid(True, alpha=0.3, which='both')
            ax4.set_xlim(xl, xr)
            
            plt.tight_layout()
            st.pyplot(fig2)
            create_download_button(fig2, f"error_profiles_n{selected_result['n']}", key="dl_fig2")
            plt.close(fig2)
            
            # Row 3: Convergence analysis
            st.markdown("#### Convergence Analysis")
            st.caption(f"Evaluation grid: n_eval = 2 × n_max = {2 * params['n_max']} points")
            
            fig3 = plt.figure(figsize=(18, 6))
            ax5 = fig3.add_subplot(1, 1, 1)
            grid_sizes = [r['n'] for r in results]
            rel_errors = [r['max_rel_error'] for r in results]
            rel_errors_ext = [r['max_rel_error_extended'] for r in results]
            
            ax5.loglog(grid_sizes, rel_errors, 'bo-', linewidth=2, markersize=8, label='Relative error')
            ax5.loglog(grid_sizes, rel_errors_ext, 'rs--', linewidth=2, markersize=8, label='Relative error (extended)')
            
            if len(grid_sizes) >= 3:
                log_n = np.log(grid_sizes[-3:])
                log_err = np.log(rel_errors[-3:])
                if all(np.isfinite(log_err)):
                    slope = np.polyfit(log_n, log_err, 1)[0]
                    ax5.plot(grid_sizes, rel_errors[0] * (np.array(grid_sizes) / grid_sizes[0])**slope,
                            '--', color='gray', alpha=0.8, linewidth=2.5, label=f'Slope ≈ {slope:.2f}')
            
            ax5.set_xlabel('Grid size (n)', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Relative error', fontsize=14, fontweight='bold')
            ax5.set_title('Convergence of Fourier Interpolation', fontsize=16, fontweight='bold')
            ax5.legend(loc='best', fontsize=11)
            ax5.grid(True, alpha=0.3, which='both')
            
            plt.tight_layout()
            st.pyplot(fig3)
            create_download_button(fig3, "convergence_plot", key="dl_fig3_plot")
            
            # Convergence table
            st.markdown("#### Convergence Results Table")
            
            rates = []
            for i in range(len(rel_errors) - 1):
                if rel_errors[i] > 0 and rel_errors[i+1] > 0:
                    rate = np.log(rel_errors[i] / rel_errors[i+1]) / np.log(2)
                    rates.append(rate)
                else:
                    rates.append(np.nan)
            
            rates_ext = []
            for i in range(len(rel_errors_ext) - 1):
                if rel_errors_ext[i] > 0 and rel_errors_ext[i+1] > 0:
                    rate = np.log(rel_errors_ext[i] / rel_errors_ext[i+1]) / np.log(2)
                    rates_ext.append(rate)
                else:
                    rates_ext.append(np.nan)
            
            table_data = []
            headers = ['Grid Size (n)', 'Extension (c)', 'Max Abs Error', 'Rel Error', 'Rel Error (ext)', 'Rate', 'Rate (ext)']
            
            for i, res in enumerate(results):
                row = [
                    f"{res['n']}",
                    f"{res['c']}",
                    f"{res['max_abs_error']:.2e}",
                    f"{res['max_rel_error']:.2e}",
                    f"{res['max_rel_error_extended']:.2e}",
                    f"{rates[i]:.2f}" if i < len(rates) and not np.isnan(rates[i]) else "—",
                    f"{rates_ext[i]:.2f}" if i < len(rates_ext) and not np.isnan(rates_ext[i]) else "—"
                ]
                table_data.append(row)
            
            n_rows = len(table_data)
            table_height = max(3, 1 + n_rows * 0.5)
            
            fig_table = plt.figure(figsize=(18, table_height))
            ax_table = fig_table.add_subplot(1, 1, 1)
            ax_table.axis('off')
            
            table = ax_table.table(cellText=table_data, colLabels=headers,
                                 cellLoc='center', loc='center',
                                 colWidths=[0.13, 0.13, 0.15, 0.15, 0.15, 0.11, 0.11])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.0)
            
            for i in range(len(headers)):
                cell = table[(0, i)]
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            
            for i in range(len(table_data)):
                for j in range(len(headers)):
                    cell = table[(i+1, j)]
                    if i % 2 == 0:
                        cell.set_facecolor('#E7E6E6')
            
            plt.tight_layout()
            st.pyplot(fig_table)
            create_download_button(fig_table, "convergence_table", key="dl_fig_table")
            plt.close(fig_table)
            plt.close(fig3)
            
            # Export buttons
            st.markdown("#### Export Convergence Data")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                csv_buffer = io.StringIO()
                csv_buffer.write("Grid_Size_n,Extension_c,Max_Abs_Error,Rel_Error,Rel_Error_ext,Rate,Rate_ext\n")
                
                for i, res in enumerate(results):
                    rate_str = f"{rates[i]:.6f}" if i < len(rates) and not np.isnan(rates[i]) else ""
                    rate_ext_str = f"{rates_ext[i]:.6f}" if i < len(rates_ext) and not np.isnan(rates_ext[i]) else ""
                    csv_buffer.write(f"{res['n']},{res['c']},{res['max_abs_error']:.6e}," +
                                    f"{res['max_rel_error']:.6e},{res['max_rel_error_extended']:.6e}," +
                                    f"{rate_str},{rate_ext_str}\n")
                
                st.download_button(
                    label="Download Table (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="convergence_table.csv",
                    mime="text/csv",
                    key="download_convergence_csv",
                    use_container_width=True
                )
            
            st.success(f"Configuration tested successfully! Ready for comparison.")
            
        else:
            st.info("Click **Run Analysis** above to test your configuration before comparing.")
    
    # TAB 2: COMPARE
    with tab_compare:
        compare_tab()
    
    # TAB 3: OSCILLATORY INTEGRATION
    with tab_oscillatory:
        oscillatory_integration_tab(app)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <small>
    <b>HighFIE Lab</b> - High-Order Fourier Interpolation with Extension<br>
    Developed at <b>Indian Institute of Technology Kanpur</b>
    </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
