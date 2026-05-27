"""Excel import/export utilities for SectionAnalyzer Pro."""

from __future__ import annotations

import pandas as pd
from io import BytesIO
from typing import Optional


def import_from_excel(uploaded_file) -> Optional[pd.DataFrame]:
    """Import data from uploaded Excel or CSV file.

    Args:
        uploaded_file: A Streamlit UploadedFile object or file-like object.

    Returns:
        DataFrame with imported data, or None on failure.
    """
    try:
        if uploaded_file is None:
            return None
        fname = getattr(uploaded_file, "name", "")
        if fname.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception:
        return None


def export_to_excel(
    data: pd.DataFrame,
    extra_sheets: Optional[dict[str, pd.DataFrame]] = None,
    filename: str = "resultados.xlsx",
) -> BytesIO:
    """Export DataFrame(s) to an Excel file in memory.

    Args:
        data: Primary DataFrame for first sheet
        extra_sheets: Dict of {sheet_name: DataFrame} for additional sheets
        filename: Output filename (unused but kept for API consistency)

    Returns:
        BytesIO buffer with .xlsx content
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        data.to_excel(writer, sheet_name="Resultados", index=False)

        if extra_sheets:
            for sheet_name, df in extra_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        # Auto-adjust column widths
        sheet_map = {"Resultados": data}
        if extra_sheets:
            sheet_map.update(extra_sheets)
        for sheet_name, sheet_df in sheet_map.items():
            if sheet_name not in writer.sheets:
                continue
            ws = writer.sheets[sheet_name]
            for i, col in enumerate(sheet_df.columns):
                col_len = max(len(str(col)), sheet_df[col].astype(str).str.len().max() if len(sheet_df) else 0)
                ws.set_column(i, i, min(col_len + 2, 30))

    output.seek(0)
    return output


def generate_template() -> BytesIO:
    """Generate a template Excel file with sample columns for batch input.

    Returns:
        BytesIO buffer with .xlsx template
    """
    template = pd.DataFrame({
        "ID": ["P1"],
        "Tipo": ["Rectangular"],
        "b (mm)": [300],
        "h (mm)": [500],
        "D (mm)": [0],
        "fc (MPa)": [25],
        "fy (MPa)": [420],
        "As (mm2)": [1600],
        "P (kN)": [1500],
        "Mx (kNm)": [200],
        "My (kNm)": [50],
    })
    return export_to_excel(template, filename="plantilla_batch.xlsx")
