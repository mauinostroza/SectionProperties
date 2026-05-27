"""
SectionAnalyzer Pro - Punto de entrada principal.

Arquitectura de layout:
┌────────────────────────────────────────────────────────────────────────────┐
│  HEADER (50px)  [Logo]  [Proyecto]  [Unidades]  [💾]  [📄]  [⚙️]           │
├──────┬──────────────────────────────────────────┬──────────────────────────┤
│ NAV  │         CANVAS CENTRAL                   │   PANEL DE               │
│ (8%) │         (60% ancho)                      │   PROPIEDADES            │
│      │                                          │   (30% ancho)            │
│ [🔲] │    • Visualización 2D/3D                 │   • Formularios          │
│ [📊] │    • Gráficos interactivos                │   • Tablas               │
│ [⚡] │    • Diagramas M-N, M-κ                   │   • Sliders              │
│      │    • Deformadas y tensiones               │   • Selectores           │
├──────┴──────────────────────────────────────────┴──────────────────────────┤
│  FOOTER (30px)  [Estado: OK]  [⏱️ Cálculo: 0.4s]                         │
└────────────────────────────────────────────────────────────────────────────┘
"""

import streamlit as st

# ── Page config MUST be first ──
st.set_page_config(
    page_title="SectionAnalyzer Pro",
    page_icon="🔲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Initialize session state ──
from state_manager import init_session_state
init_session_state()

# ── Import modules ──
from components.header import render_header
from components.sidebar_nav import render_navigation
from modules.section_module import module_section
from modules.analysis_module import module_analysis
from modules.batch_module import module_batch

# =============================================================================
# CSS PERSONALIZADO
# =============================================================================
st.markdown("""
<style>
    /* Reset padding */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* Main layout columns */
    div[data-testid="column"] {
        transition: all 0.1s ease;
    }

    /* Navigation column */
    div[data-testid="column"]:first-child {
        min-width: 80px;
        max-width: 100px;
    }
    div[data-testid="column"]:first-child .stButton button {
        font-size: 1.1rem;
        padding: 12px 4px;
        border-radius: 10px;
        margin-bottom: 6px;
        transition: all 0.15s ease;
    }
    div[data-testid="column"]:first-child .stButton button[kind="primary"] {
        background: #0d6efd;
        color: white;
        border: none;
        box-shadow: 0 2px 8px rgba(13,110,253,0.3);
    }
    div[data-testid="column"]:first-child .stButton button[kind="secondary"] {
        background: transparent;
        border: 1px solid #dee2e6;
        color: #6c757d;
    }
    div[data-testid="column"]:first-child .stButton button[kind="secondary"]:hover {
        background: #f0f2f5;
        border-color: #0d6efd;
        color: #0d6efd;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #f8f9fa;
        border-radius: 10px;
        padding: 4px;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 34px;
        padding-left: 12px;
        padding-right: 12px;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.15s ease;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }

    /* Cards and borders */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 10px;
    }
    div.stDownloadButton button, div.stButton button {
        border-radius: 8px;
    }

    /* Metrics */
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 8px 12px;
        border-left: 4px solid #0d6efd;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Data editor */
    div[data-testid="stDataEditor"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #1a1a2e;
    }

    /* Dividers */
    hr {
        margin: 0.5rem 0;
        border-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FOOTER
# =============================================================================
def render_footer() -> None:
    """Render the bottom status bar."""
    st.divider()
    f_cols = st.columns([2, 2, 1])

    mesh_info = "—"
    calc_time = "—"
    res = st.session_state.get("analysis_results", {})
    if res:
        n_elems = res.get("elements", 0)
        mesh_info = f"{n_elems:,} elementos" if n_elems else "—"

    b = st.session_state.get("b", 300)
    h = st.session_state.get("h", 500)
    section_label = f"{st.session_state.get('section_type', 'Rectangular')} {b:.0f}×{h:.0f} mm"

    f_cols[0].caption(f"🟢 Mesh: {mesh_info}")
    f_cols[1].caption(f"📐 Sección: {section_label}")
    f_cols[2].caption(f"🔲 SectionAnalyzer Pro v2.0")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    """Application entry point."""
    render_header()

    # Three-column layout: Navigation | Canvas | Properties
    nav_col, canvas_col, props_col = st.columns([0.08, 0.60, 0.30], gap="small")

    with nav_col:
        render_navigation()

    with canvas_col:
        active = st.session_state.get("active_module", "🔲 Sección")

        if active == "🔲 Sección":
            module_section()
        elif active == "📊 Análisis":
            module_analysis()
        elif active == "⚡ Batch":
            module_batch()

    with props_col:
        # Render properties panel for the active module
        active = st.session_state.get("active_module", "🔲 Sección")
        if active == "🔲 Sección":
            from modules.section_module import render_section_properties_panel
            render_section_properties_panel()
        elif active == "📊 Análisis":
            from modules.analysis_module import render_analysis_properties_panel
            render_analysis_properties_panel()
        elif active == "⚡ Batch":
            from modules.batch_module import render_batch_properties_panel
            render_batch_properties_panel()

    render_footer()


if __name__ == "__main__":
    main()
