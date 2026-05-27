"""Rebar editor component with st.data_editor, pattern generators, and import."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from config import DEFAULT_REBAR_LAYERS, DEFAULT_REBAR_MATERIALS


def render_rebar_editor(key_prefix: str = "") -> pd.DataFrame:
    """Render a data editor for reinforcement bars with pattern generators.

    Args:
        key_prefix: Unique prefix for Streamlit widget keys

    Returns:
        Edited DataFrame with rebar data
    """
    st.markdown("### Editor de refuerzo")
    st.caption("Edite directamente, pegue desde Excel, o use los generadores de patrón.")

    default_df = st.session_state.get("rebar_table", _make_default_rebar())

    r1, r2 = st.columns([3, 1])

    with r1:
        edited_df = st.data_editor(
            default_df,
            num_rows="dynamic",
            use_container_width=True,
            key=f"{key_prefix}rebar_editor",
            column_config={
                "Barra #": st.column_config.NumberColumn(
                    "Barra #", required=True, min_value=1
                ),
                "Diámetro (mm)": st.column_config.NumberColumn(
                    "Ø (mm)", min_value=6, max_value=40, step=2
                ),
                "X (mm)": st.column_config.NumberColumn(
                    "X (mm)", step=5
                ),
                "Y (mm)": st.column_config.NumberColumn(
                    "Y (mm)", step=5
                ),
                "Capa": st.column_config.SelectboxColumn(
                    "Capa",
                    options=DEFAULT_REBAR_LAYERS,
                ),
                "Material": st.column_config.SelectboxColumn(
                    "Material",
                    options=DEFAULT_REBAR_MATERIALS,
                ),
            },
        )
        st.session_state.rebar_table = edited_df

    with r2:
        _render_pattern_generator(edited_df, key_prefix)
        st.divider()
        _render_import_section(key_prefix)
        st.divider()
        _render_metrics(edited_df)

    return edited_df


def _render_pattern_generator(df: pd.DataFrame, key_prefix: str) -> None:
    """Render pattern generator controls in the side panel."""
    st.markdown("#### Generar patrón")
    pat = st.selectbox(
        "Tipo",
        ["Línea", "Rectangular", "Circular"],
        key=f"{key_prefix}pat_type",
    )

    if pat == "Línea":
        _cover = st.session_state.get("cover", 40.0)
        _b = st.session_state.get("b", 300.0)
        n = st.number_input("N barras", value=5, min_value=2, key=f"{key_prefix}pat_n")
        x0 = st.number_input("X inicial", value=_cover, key=f"{key_prefix}pat_x0")
        x1 = st.number_input("X final", value=_b - _cover, key=f"{key_prefix}pat_x1")
        y0 = st.number_input("Y fijo", value=_cover, key=f"{key_prefix}pat_y0")

        if st.button("Generar", key=f"{key_prefix}gen_line", use_container_width=True):
            new_rows = _make_line_pattern(n, x0, x1, y0, len(df))
            combined = pd.concat([df, new_rows], ignore_index=True)
            combined["Barra #"] = range(1, len(combined) + 1)
            st.session_state.rebar_table = combined
            st.rerun()

    elif pat == "Rectangular":
        rows = st.number_input("Filas", value=2, min_value=1, key=f"{key_prefix}pat_rows")
        cols = st.number_input("Columnas", value=3, min_value=1, key=f"{key_prefix}pat_cols")
        dia = st.number_input("Ø (mm)", value=16, min_value=6, max_value=40, key=f"{key_prefix}pat_dia_rect")

        if st.button("Generar", key=f"{key_prefix}gen_rect", use_container_width=True):
            new_rows = _make_rect_pattern(rows, cols, dia, len(df),
                                          st.session_state.get("b", 300),
                                          st.session_state.get("h", 500),
                                          st.session_state.get("cover", 40))
            combined = pd.concat([df, new_rows], ignore_index=True)
            combined["Barra #"] = range(1, len(combined) + 1)
            st.session_state.rebar_table = combined
            st.rerun()

    elif pat == "Circular":
        n = st.number_input("N barras", value=8, min_value=4, key=f"{key_prefix}pat_n_circ")
        dia = st.number_input("Ø (mm)", value=16, min_value=6, max_value=40, key=f"{key_prefix}pat_dia_circ")

        if st.button("Generar", key=f"{key_prefix}gen_circ", use_container_width=True):
            b = st.session_state.get("b", 300)
            h = st.session_state.get("h", 500)
            cover = st.session_state.get("cover", 40)
            new_rows = _make_circ_pattern(n, dia, len(df), b, h, cover)
            combined = pd.concat([df, new_rows], ignore_index=True)
            combined["Barra #"] = range(1, len(combined) + 1)
            st.session_state.rebar_table = combined
            st.rerun()


def _render_import_section(key_prefix: str) -> None:
    """Render file upload and clipboard import controls."""
    st.markdown("#### Importar")
    uploaded = st.file_uploader(
        "Excel/CSV",
        type=["xlsx", "csv"],
        label_visibility="collapsed",
        key=f"{key_prefix}rebar_upload",
    )
    if uploaded:
        try:
            from utils.excel_handler import import_from_excel
            df_up = import_from_excel(uploaded)
            if df_up is not None:
                st.session_state.rebar_table = df_up
                st.toast(f"Importadas {len(df_up)} barras")
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    st.button(
        "📋 Pegar desde portapapeles",
        use_container_width=True,
        key=f"{key_prefix}paste_clip",
    )


def _render_metrics(df: pd.DataFrame) -> None:
    """Render total area and reinforcement ratio metrics."""
    if len(df) > 0:
        avg_dia = df["Diámetro (mm)"].mean() if "Diámetro (mm)" in df.columns else 16
        n_bars = len(df)
        as_total = n_bars * np.pi * (avg_dia / 2) ** 2 / 1e2  # cm²
        b = st.session_state.get("b", 300)
        h = st.session_state.get("h", 500)
        rho = as_total / (b * h / 1e2) * 100  # percentage

        st.metric("As total", f"{as_total:.1f} cm²")
        st.metric("Cuantía ρ", f"{rho:.2f}%")


def _make_default_rebar() -> pd.DataFrame:
    """Return a default 4-corner-bar DataFrame."""
    return pd.DataFrame({
        "Barra #": [1, 2, 3, 4],
        "Diámetro (mm)": [16.0, 16.0, 16.0, 16.0],
        "X (mm)": [40.0, 260.0, 40.0, 260.0],
        "Y (mm)": [40.0, 40.0, 460.0, 460.0],
        "Capa": ["Ext. inf", "Ext. inf", "Ext. sup", "Ext. sup"],
        "Material": ["B500B", "B500B", "B500B", "B500B"],
    })


def _make_line_pattern(
    n: int, x0: float, x1: float, y0: float, offset: int
) -> pd.DataFrame:
    """Generate equally-spaced bars along a horizontal line."""
    xs = np.linspace(x0, x1, n)
    return pd.DataFrame({
        "Barra #": range(offset + 1, offset + n + 1),
        "Diámetro (mm)": [16.0] * n,
        "X (mm)": xs,
        "Y (mm)": [y0] * n,
        "Capa": ["Ext. inf"] * n,
        "Material": ["B500B"] * n,
    })


def _make_rect_pattern(
    rows: int, cols: int, dia: float, offset: int,
    b: float, h: float, cover: float,
) -> pd.DataFrame:
    """Generate a rectangular grid of bars."""
    n = rows * cols
    xs = np.linspace(cover, b - cover, cols)
    ys = np.linspace(cover, h - cover, rows)
    X, Y = np.meshgrid(xs, ys)

    return pd.DataFrame({
        "Barra #": range(offset + 1, offset + n + 1),
        "Diámetro (mm)": [float(dia)] * n,
        "X (mm)": X.flatten(),
        "Y (mm)": Y.flatten(),
        "Capa": ["Media"] * n,
        "Material": ["B500B"] * n,
    })


def _make_circ_pattern(
    n: int, dia: float, offset: int,
    b: float, h: float, cover: float,
) -> pd.DataFrame:
    """Generate bars in a circular arrangement."""
    radius = min(b, h) / 2 - cover
    cx, cy = b / 2, h / 2
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    return pd.DataFrame({
        "Barra #": range(offset + 1, offset + n + 1),
        "Diámetro (mm)": [float(dia)] * n,
        "X (mm)": cx + radius * np.cos(angles),
        "Y (mm)": cy + radius * np.sin(angles),
        "Capa": ["Estribo"] * n,
        "Material": ["B500B"] * n,
    })
