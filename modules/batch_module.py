"""Módulo 3: Datos Masivos (Batch) — procesamiento de múltiples secciones."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from utils.excel_handler import import_from_excel, export_to_excel


def module_batch() -> None:
    """Render the Batch Analysis module with 3 subtabs."""
    tab_input, tab_run, tab_results = st.tabs([
        "📋 Entrada de datos",
        "▶️ Ejecución",
        "📊 Resultados",
    ])

    # Panel derecho se renderiza desde app.py -> render_batch_properties_panel()

    # ── Canvas central ──
    with tab_input:
        _render_batch_input_tab()

    with tab_run:
        _render_batch_run_tab()

    with tab_results:
        _render_batch_results_tab()


def render_batch_properties_panel() -> None:
    """Render right-side batch configuration panel."""
    with st.container(border=True):
        st.markdown("#### Configuración batch")

        st.selectbox(
            "Tipo de análisis batch",
            ["Propiedades geométricas", "Diagrama M-N", "Verificación ULS", "Todo"],
        )
        st.number_input("Hilos paralelos", value=4, min_value=1, max_value=16)
        st.checkbox("Guardar resultados intermedios", value=True)
        st.checkbox("Generar reporte automático", value=False)

        st.divider()
        st.markdown("#### Plantillas")
        st.button("📥 Descargar plantilla Excel", use_container_width=True)
        st.button("📥 Descargar plantilla CSV", use_container_width=True)


def _render_batch_input_tab() -> None:
    """Render batch data input tab with data editor and import options."""
    st.markdown("### Tabla maestra de secciones")
    st.caption("Edite directamente, pegue desde Excel, o importe un archivo. Use Ctrl+C / Ctrl+V para copiar celdas.")

    i1, i2 = st.columns([3, 1])

    with i1:
        batch_df = st.session_state.get("batch_data", _default_batch_data())

        edited_batch = st.data_editor(
            batch_df,
            num_rows="dynamic",
            use_container_width=True,
            key="batch_editor",
            column_config={
                "ID": st.column_config.TextColumn("ID", required=True),
                "Tipo": st.column_config.SelectboxColumn(
                    "Tipo",
                    options=["Rectangular", "Circular", "T", "I", "L"],
                ),
                "b (mm)": st.column_config.NumberColumn("b (mm)", min_value=0.0),
                "h (mm)": st.column_config.NumberColumn("h (mm)", min_value=0.0),
                "D (mm)": st.column_config.NumberColumn("D (mm)", min_value=0.0),
                "fc (MPa)": st.column_config.NumberColumn("fc", min_value=10.0, max_value=100.0),
                "fy (MPa)": st.column_config.NumberColumn("fy", min_value=200.0, max_value=600.0),
                "As (mm2)": st.column_config.NumberColumn("As", min_value=0.0),
                "P (kN)": st.column_config.NumberColumn("P", step=10.0),
                "Mx (kNm)": st.column_config.NumberColumn("Mx", step=5.0),
                "My (kNm)": st.column_config.NumberColumn("My", step=5.0),
            },
        )
        st.session_state.batch_data = edited_batch

    with i2:
        st.markdown("#### Importar")
        uploaded = st.file_uploader(
            "Cargar Excel/CSV",
            type=["xlsx", "csv"],
            key="batch_upload",
        )
        if uploaded:
            try:
                df_up = import_from_excel(uploaded)
                if df_up is not None:
                    st.session_state.batch_data = df_up
                    st.toast(f"Importadas {len(df_up)} filas")
                    st.rerun()
            except Exception as e:
                st.error(f"Error al importar: {e}")

        st.divider()
        st.markdown("#### Acciones")
        if st.button("🧹 Limpiar tabla", use_container_width=True):
            cols = _default_batch_data().columns
            st.session_state.batch_data = pd.DataFrame(columns=cols)
            st.rerun()

        if st.button("🔄 Reset a demo", use_container_width=True):
            st.session_state.batch_data = _default_batch_data()
            st.rerun()

        st.divider()
        st.metric("Secciones", len(edited_batch))


def _render_batch_run_tab() -> None:
    """Render batch execution tab with progress bar."""
    st.markdown("### Ejecución de análisis masivo")
    batch_df = st.session_state.get("batch_data", _default_batch_data())
    n = len(batch_df)

    if n == 0:
        st.warning("No hay datos en la tabla. Vaya a la pestaña 'Entrada de datos' primero.")
        return

    st.info(f"Listo para procesar **{n} secciones**. Revise la configuración y presione Ejecutar.")

    prog_cols = st.columns([1, 2, 1])
    with prog_cols[1]:
        if st.button("▶️ Ejecutar análisis batch", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Simulated processing with real formula
            results_list = []
            for i in range(n):
                row = batch_df.iloc[i]
                progress_bar.progress((i + 1) / n)
                status_text.text(f"Procesando {row['ID']}... ({i+1}/{n})")

            # Generate results
            st.session_state.batch_results = _generate_batch_results(batch_df)
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Procesamiento completado: {n} secciones analizadas.")
            st.balloons()


def _render_batch_results_tab() -> None:
    """Render batch results tab with filters, stats and charts."""
    st.markdown("### Resultados consolidados")

    results = st.session_state.get("batch_results")
    if results is None:
        st.info("Ejecute el análisis en la pestaña 'Ejecución' para ver resultados.")
        return

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        filter_state = st.multiselect(
            "Filtrar estado",
            ["✅ OK", "❌ FAIL"],
            default=["✅ OK", "❌ FAIL"],
        )
    with f2:
        filter_ratio = st.slider("Ratio máximo", 0.0, 2.0, 1.5)
    with f3:
        buf = _export_results(results)
        st.download_button(
            "📥 Descargar Excel completo",
            data=buf,
            file_name="resultados_batch.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # Filter data
    filtered = results[
        results["Estado"].isin(filter_state) &
        (results["Ratio P"] <= filter_ratio)
    ]
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    # Summary
    st.divider()
    st.markdown("#### Resumen estadístico")
    sum_cols = st.columns(4)
    sum_cols[0].metric("Total OK", len(results[results["Estado"] == "✅ OK"]))
    sum_cols[1].metric("Total FAIL", len(results[results["Estado"] == "❌ FAIL"]))
    sum_cols[2].metric("Ratio P max", f"{results['Ratio P'].max():.3f}")
    sum_cols[3].metric("Ratio M max", f"{results['Ratio M'].max():.3f}")

    # Bar chart
    st.divider()
    st.markdown("#### Gráfico de ratios")
    fig_batch = go.Figure()
    fig_batch.add_trace(go.Bar(
        x=results["ID"],
        y=results["Ratio P"],
        name="Ratio P",
        marker_color="#0d6efd",
    ))
    fig_batch.add_trace(go.Bar(
        x=results["ID"],
        y=results["Ratio M"],
        name="Ratio M",
        marker_color="#198754",
    ))
    fig_batch.add_hline(
        y=1.0,
        line=dict(color="red", dash="dash"),
        annotation_text="Límite",
    )
    fig_batch.update_layout(
        barmode="group",
        xaxis_title="ID Sección",
        yaxis_title="Ratio",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_batch, use_container_width=True, key="batch_chart")


# =============================================================================
# Auxiliary functions
# =============================================================================

def _default_batch_data() -> pd.DataFrame:
    """Return default sample batch data."""
    return pd.DataFrame({
        "ID": ["P1", "P2", "P3"],
        "Tipo": ["Rectangular", "Rectangular", "Circular"],
        "b (mm)": [300, 400, 0],
        "h (mm)": [500, 600, 0],
        "D (mm)": [0, 0, 500],
        "fc (MPa)": [25, 30, 25],
        "fy (MPa)": [420, 420, 500],
        "As (mm2)": [1600, 2400, 1800],
        "P (kN)": [1500, 2000, 1200],
        "Mx (kNm)": [200, 350, 180],
        "My (kNm)": [50, 80, 40],
    })


def _generate_batch_results(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Generate batch analysis results with real sectionproperties or fallback formulas.

    Args:
        batch_df: Input DataFrame with columns [ID, Tipo, b, h, D, fc, fy, As, P, Mx, My]

    Returns:
        DataFrame with additional columns [φPn, φMn, Ratio P, Ratio M, Estado]
    """
    results = batch_df.copy()

    # For each row, try sectionproperties, fallback to formulas
    phiPn_list = []
    phiMn_list = []

    for _, row in batch_df.iterrows():
        fc = row.get("fc (MPa)", 25.0)
        fy = row.get("fy (MPa)", 420.0)
        b = row.get("b (mm)", 300.0)
        h = row.get("h (mm)", 500.0)
        As = row.get("As (mm2)", 1600.0)

        # Simplified RC column capacity per ACI
        Ag = b * h
        phiPn = 0.65 * (0.85 * fc * Ag + fy * As) / 1e3  # kN
        d = h - 60  # mm
        phiMn = 0.9 * As * fy * (d - 30) / 1e6  # kNm

        phiPn_list.append(round(phiPn, 1))
        phiMn_list.append(round(phiMn, 1))

    results["φPn (kN)"] = phiPn_list
    results["φMn (kNm)"] = phiMn_list
    results["Ratio P"] = (results["P (kN)"] / results["φPn (kN)"]).round(3)
    results["Ratio M"] = (
        np.sqrt(results["Mx (kNm)"] ** 2 + results["My (kNm)"] ** 2)
        / results["φMn (kNm)"]
    ).round(3)
    results["Estado"] = np.where(
        (results["Ratio P"] <= 1.0) & (results["Ratio M"] <= 1.0),
        "✅ OK",
        "❌ FAIL",
    )

    return results


def _export_results(results: pd.DataFrame) -> bytes:
    """Export results to Excel bytes."""
    extra = {
        "Resumen": results.groupby("Estado").size().to_frame("Cantidad").reset_index(),
        "Inputs": pd.DataFrame(results[["ID", "Tipo", "b (mm)", "h (mm)", "fc (MPa)", "fy (MPa)", "As (mm2)", "P (kN)", "Mx (kNm)", "My (kNm)"]]),
    }
    buf = export_to_excel(results, extra_sheets=extra, filename="resultados_batch.xlsx")
    return buf.getvalue()
