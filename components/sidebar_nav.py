"""Lateral navigation bar for SectionAnalyzer Pro.
Renders 3 icon-based navigation buttons vertically."""

from __future__ import annotations

import streamlit as st


def render_navigation() -> None:
    """Render vertical icon navigation bar with 3 modules.

    Active module is highlighted with primary button style.
    """
    modules = {
        "🔲 Sección": "Definir geometría, refuerzo y materiales",
        "📊 Análisis": "Propiedades, M-N, M-κ, tensiones",
        "⚡ Batch": "Procesamiento masivo de secciones",
    }

    for mod, desc in modules.items():
        active = st.session_state.get("active_module", "🔲 Sección") == mod
        btn_type = "primary" if active else "secondary"
        if st.button(
            mod,
            help=desc,
            use_container_width=True,
            type=btn_type,
            key=f"nav_{mod}",
        ):
            st.session_state.active_module = mod
            st.rerun()
