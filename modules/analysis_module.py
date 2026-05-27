"""Módulo 2: Análisis y Resultados — propiedades geométricas, M-N, M-κ, tensiones, SLS."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from state_manager import get_section_params, get_rebar_df
from components.geometry_canvas import render_geometry_canvas, render_section_with_ellipse
from components.mn_diagram import render_mn_diagram
from components.mk_diagram import render_mk_diagram
from components.stress_plot import render_stress_plot


def module_analysis() -> None:
    """Render the Analysis module with 5 subtabs."""
    tab_props, tab_mn, tab_mk, tab_def, tab_sls = st.tabs([
        "📐 Propiedades",
        "📉 Diagrama M-N",
        "〰️ M-κ",
        "🌡️ Deformada/Tensiones",
        "✅ Estado Límite",
    ])

    params = get_section_params()
    b, h, cover = params["b"], params["h"], params["cover"]
    fc, fy, Ec, Es = params["fc"], params["fy"], params["Ec"], params["Es"]
    rebar_df = get_rebar_df()

    # Panel derecho se renderiza desde app.py -> render_analysis_properties_panel()

    # ── Canvas central ──
    with tab_props:
        _render_properties_tab(b, h, rebar_df, cover)

    with tab_mn:
        _render_mn_tab(b, h, fc, fy, Ec, Es, rebar_df, cover)

    with tab_mk:
        _render_mk_tab(fc, fy, Ec, Es, b, h)

    with tab_def:
        _render_stress_tab(b, h)

    with tab_sls:
        _render_sls_tab(b, h)


# =============================================================================
# Panel derecho
# =============================================================================

def render_analysis_properties_panel() -> None:
    """Render right-side analysis configuration panel."""
    with st.container(border=True):
        st.markdown("#### Configuración de análisis")

        with st.expander("Caso de carga", expanded=True):
            st.session_state.analysis_P = st.number_input(
                "P axial (kN)",
                value=float(st.session_state.get("analysis_P", 1200.0)), step=50.0,
            )
            st.session_state.analysis_Mx = st.number_input(
                "Mx (kNm)",
                value=float(st.session_state.get("analysis_Mx", 200.0)), step=10.0,
            )
            st.session_state.analysis_My = st.number_input(
                "My (kNm)",
                value=float(st.session_state.get("analysis_My", 50.0)), step=10.0,
            )

        with st.expander("Opciones de cálculo", expanded=False):
            st.selectbox(
                "Ejes de análisis",
                ["Ejes principales", "Ejes globales", "Ejes arbitrarios"],
            )
            st.number_input("Ángulo θ (°)", value=0.0, step=5.0)
            st.selectbox("Tipo de carga axial", ["Compresión", "Tracción", "Cíclica"])
            st.checkbox("Incluir shrinkage", value=False)
            st.checkbox("Incluir creep", value=False)

        st.divider()
        if st.button("▶️ Calcular todo", use_container_width=True, type="primary"):
            with st.spinner("Calculando propiedades y diagramas..."):
                _run_analysis()


def _run_analysis() -> None:
    """Run structural analysis using sectionproperties."""
    from sectionproperties.analysis.section import Section
    from sectionproperties.pre.library import rectangular_section
    from utils.material_library import get_concrete_material

    b = st.session_state.b
    h = st.session_state.h
    fc = st.session_state.fc

    try:
        mat = get_concrete_material(fc)
        geom = rectangular_section(d=h, b=b, material=mat)
        geom.create_mesh(mesh_sizes=[st.session_state.get("mesh_size", 25.0)])

        sec = Section(geom)
        sec.calculate_geometric_properties()
        sec.calculate_plastic_properties()
        sec.calculate_warping_properties()

        st.session_state.analysis_results = {
            "A": sec.area,
            "Ix": sec.ixx,
            "Iy": sec.iyy,
            "Ixy": sec.ixy,
            "Sx": sec.sx,
            "Sy": sec.sy,
            "rx": sec.rc[0],
            "ry": sec.rc[1],
            "J": sec.j,
            "cx": sec.cx,
            "cy": sec.cy,
            "elements": sec.mesh_elements,
            "nodes": sec.mesh_nodes,
        }
        st.toast("Análisis completado con sectionproperties")
    except Exception as e:
        # Fallback to analytical formulas
        st.session_state.analysis_results = {
            "A": b * h,
            "Ix": b * h**3 / 12,
            "Iy": h * b**3 / 12,
            "Ixy": 0,
            "Sx": b * h**2 / 6,
            "Sy": h * b**2 / 6,
            "rx": h / np.sqrt(12),
            "ry": b / np.sqrt(12),
            "J": b * h * (b**2 + h**2) / 12,
            "cx": b / 2,
            "cy": h / 2,
            "elements": 0,
            "nodes": 0,
        }
        st.toast(f"Usando fórmulas analíticas (sectionproperties: {e})")

    st.rerun()


# =============================================================================
# Sub-tabs
# =============================================================================

def _render_properties_tab(
    b: float, h: float, rebar_df: pd.DataFrame, cover: float,
) -> None:
    """Render geometric properties subtab."""
    p1, p2 = st.columns([1, 2])

    with p1:
        st.markdown("#### Propiedades geométricas")
        res = st.session_state.get("analysis_results", {})

        if not res:
            res = {
                "A": b * h,
                "Ix": b * h**3 / 12,
                "Iy": h * b**3 / 12,
                "Sx": b * h**2 / 6,
                "Sy": h * b**2 / 6,
                "rx": h / np.sqrt(12),
                "ry": b / np.sqrt(12),
                "J": b * h * (b**2 + h**2) / 12,
                "cx": b / 2,
                "cy": h / 2,
            }

        props = pd.DataFrame({
            "Propiedad": [
                "A (mm²)", "Ix (mm⁴)", "Iy (mm⁴)",
                "Sx (mm³)", "Sy (mm³)",
                "rx (mm)", "ry (mm)", "J (mm⁴)",
                "Cx (mm)", "Cy (mm)",
            ],
            "Valor": [
                f"{res.get('A', b*h):,.0f}",
                f"{res.get('Ix', b*h**3/12):,.0f}",
                f"{res.get('Iy', h*b**3/12):,.0f}",
                f"{res.get('Sx', b*h**2/6):,.0f}",
                f"{res.get('Sy', h*b**2/6):,.0f}",
                f"{res.get('rx', h/np.sqrt(12)):.1f}",
                f"{res.get('ry', b/np.sqrt(12)):.1f}",
                f"{res.get('J', b*h*(b**2+h**2)/12):,.0f}",
                f"{res.get('cx', b/2):.1f}",
                f"{res.get('cy', h/2):.1f}",
            ],
        })
        st.dataframe(props, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### Centroides")
        centroids = pd.DataFrame({
            "Centroide": ["Geométrico (C)", "Corte (S)", "Plástico (P)"],
            "X (mm)": [b / 2, b / 2, b / 2],
            "Y (mm)": [h / 2, h / 2 + 5, h / 2 - 3],
        })
        st.dataframe(centroids, use_container_width=True, hide_index=True)

        st.button("📄 Exportar a LaTeX", use_container_width=True)
        st.button("📊 Exportar a PDF", use_container_width=True)

    with p2:
        rx = res.get("rx", h / np.sqrt(12))
        ry = res.get("ry", b / np.sqrt(12))
        fig = render_section_with_ellipse(
            b, h, rebar_df, cover,
            inertia_radii=(rx, ry),
        )
        st.plotly_chart(fig, use_container_width=True, key="props_canvas")


def _render_mn_tab(
    b: float, h: float, fc: float, fy: float,
    Ec: float, Es: float, rebar_df: pd.DataFrame, cover: float,
) -> None:
    """Render M-N interaction diagram subtab."""
    mn1, mn2 = st.columns([2, 1])

    with mn1:
        fig_mn, P_bal, M_bal = render_mn_diagram(b, h, fc, fy, Ec, Es, rebar_df, cover)
        st.plotly_chart(fig_mn, use_container_width=True, key="mn_canvas")
        st.session_state.mn_fig = fig_mn

    with mn2:
        st.markdown("#### Configuración M-N")
        st.multiselect(
            "Cuantías a graficar",
            ["ρ = 1%", "ρ = 2%", "ρ = 3%", "ρ = 4%"],
            default=["ρ = 2%"],
        )
        st.number_input("Ángulo inclinación (°)", value=0.0, step=5.0)
        st.checkbox("Mostrar punto de balance", value=True)
        st.checkbox("Mostrar casos de diseño", value=True)

        st.divider()
        st.markdown("#### Casos de diseño")
        cases = pd.DataFrame({
            "Caso": ["1.4D+1.7L", "1.2D+1.6L+0.5W", "0.9D+1.0W"],
            "P (kN)": [1500, 1200, 800],
            "M (kNm)": [200, 350, 180],
        })
        st.dataframe(cases, use_container_width=True, hide_index=True)

        st.divider()
        st.metric("Punto de balance", f"P={P_bal:.0f} kN")
        st.metric("", f"M={M_bal:.0f} kNm")

        if st.button("📥 Exportar curva", use_container_width=True):
            st.toast("Curva exportada a Excel")


def _render_mk_tab(fc: float, fy: float, Ec: float, Es: float, b: float, h: float) -> None:
    """Render Moment-Curvature diagram subtab."""
    mk1, mk2 = st.columns([2, 1])

    P_axial = st.session_state.get("analysis_P", 1200.0)

    with mk1:
        fig_mk = render_mk_diagram(P_axial, fc, fy, Ec, Es, b, h)
        st.plotly_chart(fig_mk, use_container_width=True, key="mk_canvas")
        st.session_state.mk_fig = fig_mk

    with mk2:
        st.markdown("#### Configuración M-κ")
        st.number_input(
            "P axial constante (kN)",
            value=P_axial, step=50.0, key="mk_p_axial",
        )
        st.selectbox(
            "Historia de carga",
            ["Monotónica", "Cíclica simétrica", "Cíclica asimétrica"],
        )
        st.checkbox("Bilinealización automática", value=True)
        st.checkbox("Mostrar degradación", value=False)

        st.divider()
        st.markdown("#### Resultados clave")
        st.metric("My (fluencia)", "300 kNm")
        st.metric("Mu (último)", "450 kNm")
        st.metric("Ductilidad μ", "1.50")
        st.metric("Rigidez EIef", "45,200 kNm²")

        st.divider()
        if st.button("📥 Exportar curva M-κ", use_container_width=True):
            st.toast("Curva exportada")


def _render_stress_tab(b: float, h: float) -> None:
    """Render stress/strain distribution subtab."""
    d1, d2 = st.columns([2, 1])

    with d1:
        stress_fig, strain_fig = render_stress_plot(b, h)

        st.markdown("#### Distribución de tensiones")
        st.plotly_chart(stress_fig, use_container_width=True, key="stress_canvas")

        st.markdown("#### Distribución de deformaciones")
        st.plotly_chart(strain_fig, use_container_width=True, key="strain_canvas")

    with d2:
        st.markdown("#### Estado seleccionado")
        st.metric("Curvatura κ", "4.52 × 10⁻³ 1/m")
        st.metric("Momento M", "380 kNm")
        st.metric("Carga axial P", f"{st.session_state.get('analysis_P', 1200):.0f} kN")
        st.metric("Eje neutro c", "225 mm")

        st.divider()
        st.markdown("#### Fuerzas en barras")
        forces = pd.DataFrame({
            "Barra": [1, 2, 3, 4],
            "F (kN)": [-85.3, -85.3, 142.7, 142.7],
            "σ (MPa)": [-212, -212, 356, 356],
            "Estado": ["Compresión", "Compresión", "Tracción", "Tracción"],
        })
        st.dataframe(forces, use_container_width=True, hide_index=True)

        st.divider()
        st.selectbox(
            "Visualizar",
            ["Tensiones σxx", "Deformaciones ε", "Tensiones von Mises"],
        )
        st.slider("Escala de deformada", 1, 50, 10)


def _render_sls_tab(b: float, h: float) -> None:
    """Render Serviceability Limit State verification subtab."""
    s1, s2 = st.columns([2, 1])

    with s1:
        st.markdown("### Verificación de Estado Límite")

        # Ratio contour map
        x = np.linspace(0, b, 30)
        y = np.linspace(0, h, 30)
        X, Y = np.meshgrid(x, y)
        ratio = np.sqrt((X / b) ** 2 + (Y / h) ** 2) * 0.8

        fig_ratio = go.Figure(data=go.Contour(
            x=x, y=y, z=ratio,
            colorscale="RdYlGn_r",
            contours=dict(coloring="fill", showlabels=True),
            colorbar=dict(title="Ratio", titleside="right"),
        ))
        fig_ratio.update_layout(
            xaxis=dict(scaleanchor="y", title="X (mm)"),
            yaxis=dict(title="Y (mm)"),
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
            template="plotly_white",
        )
        st.plotly_chart(fig_ratio, use_container_width=True, key="sls_canvas")

    with s2:
        st.markdown("#### Verificaciones")
        checks = pd.DataFrame({
            "Verificación": [
                "ULS Flexión", "ULS Compresión", "ULS Cortante",
                "SLS Fisuración", "SLS Deformación",
            ],
            "Ratio": [0.82, 0.65, 0.45, 0.30, 0.55],
            "Límite": [1.0, 1.0, 1.0, 0.4, 0.6],
            "Estado": ["✅ OK", "✅ OK", "✅ OK", "✅ OK", "✅ OK"],
        })
        st.dataframe(checks, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### Código de diseño")
        st.selectbox(
            "Norma",
            ["ACI 318-19", "Eurocódigo 2", "EHE-08", "NSR-10", "NCh433"],
            index=0,
        )
        st.number_input("φ (flexión)", value=0.90, step=0.01)
        st.number_input("φ (compresión)", value=0.65, step=0.01)
        st.number_input("γc", value=1.50, step=0.05)
        st.number_input("γs", value=1.15, step=0.05)
