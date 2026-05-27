"""Canvas renderer for 2D section visualization with reinforcement."""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.plotly_themes import SECTION_COLORS


def render_geometry_canvas(
    b: float,
    h: float,
    rebar_df: pd.DataFrame | None = None,
    cover: float = 40.0,
    show_centroids: bool = True,
    height: int = 450,
    theme: str = "Claro",
) -> go.Figure:
    """Generate plotly figure of a rectangular section with rebar markers.

    Args:
        b: Section width in mm
        h: Section height in mm
        rebar_df: DataFrame with columns [Barra#, Diámetro (mm), X (mm), Y (mm), Capa, Material]
        cover: Cover in mm (for visual reference)
        show_centroids: Show centroidal axes
        height: Figure height in px
        theme: Theme name for styling

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Concrete section outline
    fig.add_trace(go.Scatter(
        x=[0, b, b, 0, 0],
        y=[0, 0, h, h, 0],
        fill="toself",
        fillcolor=SECTION_COLORS["concrete"],
        line=dict(color=SECTION_COLORS["concrete_stroke"], width=2),
        name="Hormigón",
        hoverinfo="skip",
        showlegend=False,
    ))

    # Centroidal axes
    if show_centroids:
        fig.add_hline(
            y=h / 2,
            line=dict(color=SECTION_COLORS["centroidal"], width=1, dash="dash"),
            opacity=0.5,
        )
        fig.add_vline(
            x=b / 2,
            line=dict(color=SECTION_COLORS["centroidal"], width=1, dash="dash"),
            opacity=0.5,
        )

    # Reinforcement bars
    if rebar_df is not None and len(rebar_df) > 0:
        layer_colors = {
            "Ext. inf": SECTION_COLORS["rebar_tension"],
            "Ext. sup": SECTION_COLORS["rebar_compression"],
            "Media": SECTION_COLORS["rebar_stirrup"],
            "Estribo": SECTION_COLORS["rebar_stirrup"],
        }
        default_color = SECTION_COLORS["rebar_tension"]

        for _, row in rebar_df.iterrows():
            dia = float(row.get("Diámetro (mm)", 16.0))
            x = float(row.get("X (mm)", 0.0))
            y = float(row.get("Y (mm)", 0.0))
            capa = str(row.get("Capa", "General"))
            color = layer_colors.get(capa, default_color)
            bar_num = int(row.get("Barra #", 0))

            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker=dict(
                    size=max(dia / 1.5, 4),
                    color=color,
                    line=dict(color="black", width=1),
                ),
                name=f"Ø{dia:.0f}",
                hovertemplate=(
                    f"Barra #{bar_num}<br>"
                    f"X: {x:.1f}<br>"
                    f"Y: {y:.1f}<br>"
                    f"Ø: {dia:.0f} mm"
                    f"<extra></extra>"
                ),
                showlegend=False,
            ))

    # Layout
    padding = max(b, h) * 0.15
    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            title="X (mm)",
            range=[-padding, b + padding],
        ),
        yaxis=dict(
            title="Y (mm)",
            range=[-padding, h + padding],
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=height,
        template="plotly_white",
        dragmode="pan",
    )

    return fig


def render_section_with_ellipse(
    b: float,
    h: float,
    rebar_df: pd.DataFrame | None = None,
    cover: float = 40.0,
    inertia_radii: tuple[float, float] | None = None,
) -> go.Figure:
    """Render section with inertia ellipse overlay.

    Args:
        b, h: Section dimensions
        rebar_df: Reinforcement DataFrame
        cover: Cover depth
        inertia_radii: (rx, ry) radii of gyration for ellipse

    Returns:
        Plotly Figure with ellipse
    """
    fig = render_geometry_canvas(b, h, rebar_df, cover, show_centroids=True)

    if inertia_radii:
        rx, ry = inertia_radii
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=b / 2 + rx * np.cos(theta),
            y=h / 2 + ry * np.sin(theta),
            mode="lines",
            line=dict(color=SECTION_COLORS["primary"], dash="dash", width=2),
            name="Elipse de inercia",
            hoverinfo="skip",
            showlegend=False,
        ))

    return fig
