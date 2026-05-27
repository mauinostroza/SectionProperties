"""Stress and strain contour plot components."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from utils.plotly_themes import SECTION_COLORS, get_contour_colorscale


def render_stress_plot(
    b: float,
    h: float,
    section_props: dict | None = None,
    height: int = 450,
) -> tuple[go.Figure, go.Figure]:
    """Generate stress contour and strain distribution plots.

    Args:
        b: Section width in mm
        h: Section height in mm
        section_props: Dict with analysis results (optional, for real data)
        height: Figure height in px

    Returns:
        (stress_fig, strain_fig) tuple
    """
    stress_fig = _mock_stress_contour(b, h, height)
    strain_fig = _mock_strain_distribution(b, h, height)
    return stress_fig, strain_fig


def _mock_stress_contour(b: float, h: float, height: int) -> go.Figure:
    """Generate mock stress contour with parabolic compression distribution."""
    nx, ny = 50, 50
    x = np.linspace(0, b, nx)
    y = np.linspace(0, h, ny)
    X, Y = np.meshgrid(x, y)

    # Parabolic compression with neutral axis
    neutral_axis_y = h * 0.45
    Z = np.where(
        Y > neutral_axis_y,
        -25 * (1 - ((Y - neutral_axis_y) / (h - neutral_axis_y))**2),
        0.0,
    )
    # Smooth variation across width
    Z *= np.cos(np.pi * (X / b - 0.5))

    fig = go.Figure(data=go.Contour(
        x=x,
        y=y,
        z=Z,
        colorscale=get_contour_colorscale("stress"),
        contours=dict(coloring="fill", showlabels=True),
        colorbar=dict(title="σ (MPa)", titleside="right"),
        hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}<br>σ: %{z:.2f} MPa<extra></extra>",
    ))

    # Neutral axis line
    fig.add_hline(
        y=neutral_axis_y,
        line=dict(color=SECTION_COLORS["neutral_axis"], width=2, dash="dot"),
        annotation_text="Eje neutro",
    )

    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1, title="X (mm)"),
        yaxis=dict(title="Y (mm)"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=height,
        template="plotly_white",
    )
    return fig


def _mock_strain_distribution(b: float, h: float, height: int) -> go.Figure:
    """Generate linear strain distribution across section height."""
    neutral_axis_y = h * 0.45
    y_vals = np.linspace(0, h, 100)

    # Linear strain: compression negative, tension positive
    strain = -0.0035 * (y_vals - neutral_axis_y) / neutral_axis_y

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strain * 1000,
        y=y_vals,
        mode="lines",
        fill="tozerox",
        fillcolor="rgba(220, 53, 69, 0.2)",
        line=dict(color=SECTION_COLORS["danger"], width=3),
        name="ε",
        hovertemplate="ε: %{x:.3f} ‰<br>Y: %{y:.0f} mm<extra></extra>",
    ))

    # Zero strain axis
    fig.add_vline(
        x=0,
        line=dict(color="black", width=1),
    )

    fig.update_layout(
        xaxis_title="ε (‰)",
        yaxis_title="Y (mm)",
        margin=dict(l=40, r=20, t=20, b=40),
        height=max(250, height // 2),
        template="plotly_white",
        showlegend=False,
    )
    return fig
