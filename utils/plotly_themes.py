"""Plotly theme utilities for consistent visual styling."""

from __future__ import annotations

import plotly.graph_objects as go


SECTION_COLORS = {
    "concrete": "rgba(180, 180, 180, 0.4)",
    "concrete_stroke": "#555555",
    "rebar_tension": "#2E86AB",
    "rebar_compression": "#A23B72",
    "rebar_stirrup": "#F18F01",
    "centroidal": "red",
    "principal": "green",
    "neutral_axis": "black",
    "safe": "#198754",
    "warning": "#FFC107",
    "danger": "#DC3545",
    "primary": "#0D6EFD",
}

THEME_TEMPLATES = {
    "Claro": "plotly_white",
    "Oscuro": "plotly_dark",
    "Alto contraste": "seaborn",
}


def apply_theme(fig: go.Figure, theme: str = "Claro") -> go.Figure:
    """Apply a theme template to a plotly figure.

    Args:
        fig: Plotly figure to theme
        theme: One of 'Claro', 'Oscuro', 'Alto contraste'

    Returns:
        Themed figure
    """
    template = THEME_TEMPLATES.get(theme, "plotly_white")
    fig.update_layout(template=template)
    return fig


def get_contour_colorscale(quantity: str = "stress") -> str:
    """Return appropriate colorscale for a physical quantity.

    Args:
        quantity: 'stress', 'strain', 'ratio', 'temperature'

    Returns:
        Plotly colorscale name
    """
    scales = {
        "stress": "RdBu_r",
        "strain": "Viridis",
        "ratio": "RdYlGn_r",
        "temperature": "Inferno",
    }
    return scales.get(quantity, "Viridis")


def default_margins(height: int = 450) -> dict:
    """Return default layout margins for structural plots."""
    return dict(l=40, r=20, t=40, b=40)


def axis_config(scale_anchor: str = "y", title_x: str = "X (mm)", title_y: str = "Y (mm)") -> dict:
    """Return standard axis configuration for section plots."""
    return dict(
        xaxis=dict(scaleanchor=scale_anchor, scaleratio=1, title=title_x),
        yaxis=dict(title=title_y),
    )
