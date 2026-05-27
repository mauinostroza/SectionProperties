"""Persistent header bar for SectionAnalyzer Pro."""

from __future__ import annotations

import streamlit as st
from config import UNIT_SYSTEMS, DESIGN_CODES


def render_header() -> None:
    """Render the top header bar with project context, units, and action buttons."""
    with st.container():
        cols = st.columns([2, 3, 1.5, 1, 1, 1])

        # Logo / Title
        cols[0].markdown("### 🔲 SectionAnalyzer Pro")

        # Project name
        cols[1].text_input(
            "Proyecto",
            value=st.session_state.get("project_name", "Edificio A - Pilar P1"),
            label_visibility="collapsed",
            placeholder="Nombre del proyecto...",
            key="project_name_input",
        )

        # Unit system selector
        unit_options = list(UNIT_SYSTEMS.keys())
        current_units = st.session_state.get("units", "kN, m")
        default_idx = unit_options.index(current_units) if current_units in unit_options else 0
        units = cols[2].selectbox(
            "Unidades",
            unit_options,
            index=default_idx,
            label_visibility="collapsed",
            key="header_units",
        )
        if units != st.session_state.get("units"):
            st.session_state.units = units
            st.rerun()

        # Action buttons
        cols[3].button("💾 Guardar", use_container_width=True, key="header_save")
        cols[4].button("📄 Exportar", use_container_width=True, key="header_export")

        if cols[5].button("⚙️", use_container_width=True, key="header_settings"):
            st.session_state.settings_open = not st.session_state.settings_open

        st.divider()

        # Settings drawer (collapsible)
        if st.session_state.get("settings_open", False):
            _render_settings_drawer()
            st.divider()


def _render_settings_drawer() -> None:
    """Render the settings popover with design code, theme, and solver options."""
    with st.container(border=True):
        s_cols = st.columns(4)

        # Design code
        code_options = DESIGN_CODES
        current_code = st.session_state.get("code", "ACI 318-19")
        default_code = code_options.index(current_code) if current_code in code_options else 0
        code = s_cols[0].selectbox(
            "Código de diseño",
            code_options,
            index=default_code,
            key="settings_code",
        )
        st.session_state.code = code

        # Theme
        theme_options = ["Claro", "Oscuro", "Alto contraste"]
        current_theme = st.session_state.get("theme", "Claro")
        default_theme = theme_options.index(current_theme) if current_theme in theme_options else 0
        theme = s_cols[1].selectbox(
            "Tema",
            theme_options,
            index=default_theme,
            key="settings_theme",
        )
        st.session_state.theme = theme

        # Solver tolerance
        s_cols[2].number_input(
            "Tolerancia solver",
            value=1e-6,
            format="%.0e",
            key="solver_tol",
        )

        # Close button
        if s_cols[3].button("Cerrar", use_container_width=True):
            st.session_state.settings_open = False
            st.rerun()
