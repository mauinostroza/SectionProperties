"""Módulo 1: Sección Transversal — definición de geometría, refuerzo, materiales y malla."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from state_manager import get_section_params, get_rebar_df
from config import (
    SECTION_TYPES,
    MATERIAL_PRESETS,
    CONCRETE_MODELS,
    STEEL_MODELS,
    SOLVER_TYPES,
)
from components.geometry_canvas import render_geometry_canvas
from components.rebar_editor import render_rebar_editor


def module_section() -> None:
    """Render the Section Definition module with 5 subtabs."""
    tab_lib, tab_geo, tab_reinf, tab_mat, tab_mesh = st.tabs([
        "📚 Biblioteca",
        "📐 Geometría",
        "🔩 Refuerzo",
        "🧱 Materiales",
        "🔲 Malla",
    ])

    params = get_section_params()
    b, h, cover = params["b"], params["h"], params["cover"]
    rebar_df = get_rebar_df()

    # Panel derecho se renderiza desde app.py -> render_section_properties_panel()

    # ── Canvas central ──
    with tab_lib:
        _render_library_tab()

    with tab_geo:
        _render_geometry_tab(b, h, rebar_df, cover)

    with tab_reinf:
        _render_reinforcement_tab(b, h, cover)

    with tab_mat:
        _render_materials_tab()

    with tab_mesh:
        _render_mesh_tab(b, h)


# =============================================================================
# Panel derecho
# =============================================================================

def render_section_properties_panel() -> None:
    """Render right-side properties panel for the section module."""
    with st.container(border=True):
        st.markdown("#### Propiedades de la sección")

        with st.expander("Dimensiones", expanded=True):
            st.session_state.b = st.number_input(
                "Base b (mm)", min_value=50.0, max_value=5000.0,
                value=float(st.session_state.get("b", 300.0)), step=10.0,
            )
            st.session_state.h = st.number_input(
                "Altura h (mm)", min_value=50.0, max_value=5000.0,
                value=float(st.session_state.get("h", 500.0)), step=10.0,
            )
            st.session_state.cover = st.number_input(
                "Recubrimiento c (mm)", min_value=10.0, max_value=200.0,
                value=float(st.session_state.get("cover", 40.0)), step=5.0,
            )

        with st.expander("Materiales básicos", expanded=True):
            st.session_state.fc = st.number_input(
                "f'c (MPa)", min_value=10.0, max_value=100.0,
                value=float(st.session_state.get("fc", 25.0)), step=1.0,
            )
            st.session_state.fy = st.number_input(
                "fy (MPa)", min_value=200.0, max_value=600.0,
                value=float(st.session_state.get("fy", 420.0)), step=10.0,
            )
            st.session_state.Ec = st.number_input(
                "Ec (MPa)",
                value=float(st.session_state.get("Ec", 25000.0)), step=1000.0,
            )
            st.session_state.Es = st.number_input(
                "Es (MPa)",
                value=float(st.session_state.get("Es", 200000.0)), step=10000.0,
            )

        with st.expander("Opciones avanzadas", expanded=False):
            st.selectbox("Curva hormigón", CONCRETE_MODELS)
            st.selectbox("Curva acero", STEEL_MODELS)
            st.number_input("Deformación última εcu", value=0.0035, format="%.4f")
            st.number_input("Deformación acero εsu", value=0.09, format="%.4f")
            st.checkbox("Considerar confinamiento", value=False)
            st.checkbox("Efecto de fluencia lenta", value=False)

        st.divider()
        if st.button("🔄 Actualizar vista", use_container_width=True, type="primary"):
            st.rerun()

        area_bruta = st.session_state.b * st.session_state.h / 1e6
        st.caption(f"Área bruta: {area_bruta:.3f} m²")


# =============================================================================
# Sub-tabs
# =============================================================================

def _render_library_tab() -> None:
    """Render section library with predefined types."""
    st.markdown("### Biblioteca de secciones predefinidas")
    lib_cols = st.columns(4)

    sections_demo = [
        ("Rectangular", "📦", "300×500"),
        ("Circular", "🔵", "Ø500"),
        ("T", "⫘", "b=800, h=600"),
        ("I", "⧉", "W24×62"),
        ("L", "📐", "L150×100"),
        ("Cajón", "⬜", "600×400"),
        ("Compuesta", "🔗", "Hormigón + Acero"),
        ("Poligonal", "⬡", "Custom"),
    ]

    for i, (name, icon, dims) in enumerate(sections_demo):
        with lib_cols[i % 4]:
            with st.container(border=True):
                st.markdown(f"**{icon} {name}**")
                st.caption(dims)
                if st.button("Usar", key=f"use_{name}", use_container_width=True):
                    st.session_state.section_type = name
                    st.toast(f"Sección cambiada a {name}")
                    st.rerun()


def _render_geometry_tab(b: float, h: float, rebar_df: pd.DataFrame, cover: float) -> None:
    """Render geometry subtab with canvas and coordinate info."""
    c1, c2 = st.columns([2, 1])

    with c1:
        fig = render_geometry_canvas(b, h, rebar_df, cover)
        st.plotly_chart(fig, use_container_width=True, key="geo_canvas")

    with c2:
        st.markdown("#### Operaciones")
        st.selectbox(
            "Herramienta",
            ["Seleccionar", "Rectángulo", "Círculo", "Polígono", "Agujero", "Mover"],
        )
        st.checkbox("Snap a grid", value=True)
        st.checkbox("Mostrar centroides", value=True)
        st.checkbox("Mostrar ejes principales", value=False)

        if st.button("Centrar vista"):
            st.toast("Vista centrada")

        st.divider()
        st.markdown("#### Coordenadas")
        st.dataframe(
            pd.DataFrame({
                "Punto": ["A", "B", "C", "D"],
                "X (mm)": [0, b, b, 0],
                "Y (mm)": [0, 0, h, h],
            }),
            use_container_width=True,
            hide_index=True,
        )


def _render_reinforcement_tab(b: float, h: float, cover: float) -> None:
    """Render reinforcement editor subtab."""
    edited_df = render_rebar_editor(key_prefix="section_")

    st.divider()
    st.markdown("#### Vista previa")
    fig = render_geometry_canvas(b, h, edited_df, cover)
    st.plotly_chart(fig, use_container_width=True, key="reinf_canvas")


def _render_materials_tab() -> None:
    """Render materials subtab with concrete and steel curves."""
    m1, m2 = st.columns(2)

    with m1:
        with st.container(border=True):
            st.markdown("#### 🧱 Hormigón")
            fc = st.number_input(
                "f'c (MPa)", value=st.session_state.get("fc", 25.0),
                key="mat_fc",
            )
            st.number_input(
                "Ec (MPa)", value=st.session_state.get("Ec", 25000.0),
                key="mat_Ec",
            )
            st.number_input("γ (kN/m³)", value=24.0)
            st.number_input("ν", value=0.2, step=0.01)
            st.selectbox("Modelo", CONCRETE_MODELS)

            # σ-ε curve for concrete
            eps = np.linspace(0, 0.0035, 50)
            sigma = fc * (1 - (1 - eps / 0.002) ** 2)
            sigma = np.clip(sigma, 0, fc)
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(
                x=eps * 1000, y=sigma, fill="tozeroy", name="σ-ε",
            ))
            fig_c.update_layout(
                xaxis_title="ε (‰)", yaxis_title="σ (MPa)",
                height=250, margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_c, use_container_width=True, key="stress_concrete")

    with m2:
        with st.container(border=True):
            st.markdown("#### 🔩 Acero de refuerzo")
            fy = st.number_input(
                "fy (MPa)", value=st.session_state.get("fy", 420.0),
                key="mat_fy",
            )
            Es = st.number_input(
                "Es (MPa)", value=st.session_state.get("Es", 200000.0),
                key="mat_Es",
            )
            st.number_input("γ (kN/m³)", value=78.5)
            st.number_input("εy", value=fy / Es, format="%.5f")
            st.selectbox("Modelo", STEEL_MODELS)

            # σ-ε curve for steel
            eps_s = np.linspace(0, 0.02, 100)
            sigma_s = np.minimum(eps_s * Es, fy)
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(
                x=eps_s * 1000, y=sigma_s, fill="tozeroy", name="σ-ε",
            ))
            fig_s.update_layout(
                xaxis_title="ε (‰)", yaxis_title="σ (MPa)",
                height=250, margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_s, use_container_width=True, key="stress_steel")


def _render_mesh_tab(b: float, h: float) -> None:
    """Render mesh generation subtab."""
    mesh_cols = st.columns([2, 1])

    with mesh_cols[0]:
        # Mesh preview
        n_x = max(2, int(b / st.session_state.get("mesh_size", 25.0)))
        n_y = max(2, int(h / st.session_state.get("mesh_size", 25.0)))
        x_mesh = np.linspace(0, b, n_x)
        y_mesh = np.linspace(0, h, n_y)
        X_m, Y_m = np.meshgrid(x_mesh, y_mesh)

        fig_mesh = go.Figure()
        fig_mesh.add_trace(go.Scatter(
            x=X_m.flatten(),
            y=Y_m.flatten(),
            mode="markers",
            marker=dict(size=3, color="#0d6efd", opacity=0.6),
            name="Nodos",
            hoverinfo="skip",
        ))
        fig_mesh.add_trace(go.Scatter(
            x=[0, b, b, 0, 0],
            y=[0, 0, h, h, 0],
            mode="lines",
            line=dict(color="black", width=2),
            name="Perímetro",
            hoverinfo="skip",
        ))
        fig_mesh.update_layout(
            xaxis=dict(scaleanchor="y", title="X (mm)"),
            yaxis=dict(title="Y (mm)"),
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(fig_mesh, use_container_width=True, key="mesh_canvas")

    with mesh_cols[1]:
        st.markdown("#### Parámetros de malla")
        st.session_state.mesh_size = st.number_input(
            "Tamaño elem. (mm)",
            value=st.session_state.get("mesh_size", 25.0),
            step=5.0,
        )
        st.number_input("Ángulo máx. (°)", value=45.0, step=5.0)
        st.number_input("Calidad mín.", value=0.1, step=0.05)
        st.checkbox("Refinar en refuerzo", value=True)
        st.checkbox("Refinar en bordes", value=False)

        st.divider()
        st.markdown("#### Solver")
        st.selectbox("Tipo", SOLVER_TYPES)
        st.checkbox("Análisis paralelo", value=True)

        st.divider()
        if st.button("🔁 Generar malla", use_container_width=True, type="primary"):
            est = int((b / st.session_state.mesh_size) * (h / st.session_state.mesh_size) * 2)
            st.toast(f"Malla generada: ~{est} elementos")
            st.rerun()
        st.caption(f"Estimado: ~{int((b/st.session_state.mesh_size)*(h/st.session_state.mesh_size)*2)} elementos")
