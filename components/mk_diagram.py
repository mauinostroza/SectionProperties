"""Moment-Curvature (M-κ) diagram component."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from utils.plotly_themes import SECTION_COLORS


def render_mk_diagram(
    P_axial: float = 1200.0,
    fc: float = 25.0,
    fy: float = 420.0,
    Ec: float = 25000.0,
    Es: float = 200000.0,
    b: float = 300.0,
    h: float = 500.0,
    height: int = 450,
) -> go.Figure:
    """Generate Moment-Curvature diagram.

    Tries concreteproperties moment_curvature_diagram, falls back to mock.

    Args:
        P_axial: Constant axial load in kN
        fc: Concrete compressive strength in MPa
        fy: Steel yield strength in MPa
        Ec: Concrete elastic modulus in MPa
        Es: Steel elastic modulus in MPa
        b: Section width in mm
        h: Section height in mm
        height: Figure height in px

    Returns:
        Plotly Figure
    """
    try:
        return _real_mk_diagram(P_axial, fc, fy, Ec, Es, b, h, height)
    except Exception:
        return _mock_mk_diagram(height)


def _real_mk_diagram(
    P_axial: float, fc: float, fy: float,
    Ec: float, Es: float, b: float, h: float,
    height: int,
) -> go.Figure:
    """Real M-κ using concreteproperties MomentCurvatureDiagram."""
    from concreteproperties.material import Concrete, Steel
    from concreteproperties.stress_strain_profile import (
        RectangularStressBlock,
        SteelElasticPlastic,
    )
    from concreteproperties.concrete_section import ConcreteSection
    from sectionproperties.pre.library import rectangular_section

    concrete = Concrete(
        name="Concrete",
        density=2.4e-6,
        stress_strain_profile=RectangularStressBlock(
            compressive_strength=fc, tensile_strength=0.0, ultimate_strain=0.003,
        ),
        flexural_tensile_strength=0.0,
    )

    geom = rectangular_section(d=h, b=b, material=concrete)
    conc_sec = ConcreteSection(geom)

    from concreteproperties.results import MomentCurvatureDiagram
    mc = MomentCurvatureDiagram(conc_sec)
    mc.generate_diagram(n=P_axial * 1e3)  # convert kN to N

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[getattr(r, 'kappa', 0) * 1e3 for r in mc.results],
        y=[getattr(r, 'mx', 0) / 1e6 for r in mc.results],
        mode="lines",
        line=dict(color=SECTION_COLORS["safe"], width=3),
        name="M-κ",
    ))

    fig.update_layout(
        xaxis_title="Curvatura κ (×10⁻³ 1/m)",
        yaxis_title="Momento M (kNm)",
        margin=dict(l=40, r=20, t=40, b=40),
        height=height,
        template="plotly_white",
    )
    return fig


def _mock_mk_diagram(height: int) -> go.Figure:
    """Generate realistic mock M-κ curve for RC section."""
    kappa = np.linspace(0, 0.015, 100)
    My = 300.0   # Yield moment (kNm)
    Mu = 450.0   # Ultimate moment (kNm)
    phi_y = 0.003   # Yield curvature
    phi_u = 0.012   # Ultimate curvature

    M = np.where(
        kappa < phi_y,
        My * (kappa / phi_y),
        My + (Mu - My) * ((kappa - phi_y) / (phi_u - phi_y)) ** 0.5,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=kappa * 1000,
        y=M,
        mode="lines",
        line=dict(color=SECTION_COLORS["safe"], width=3),
        name="M-κ",
        hovertemplate="κ: %{x:.4f} ×10⁻³ 1/m<br>M: %{y:.1f} kNm<extra></extra>",
    ))

    # Yield and ultimate markers
    fig.add_vline(
        x=phi_y * 1000,
        line=dict(color=SECTION_COLORS["warning"], dash="dash"),
        annotation_text="My",
    )
    fig.add_vline(
        x=phi_u * 1000,
        line=dict(color=SECTION_COLORS["danger"], dash="dash"),
        annotation_text="Mu",
    )

    fig.update_layout(
        xaxis_title="Curvatura κ (×10⁻³ 1/m)",
        yaxis_title="Momento M (kNm)",
        margin=dict(l=40, r=20, t=40, b=40),
        height=height,
        template="plotly_white",
        showlegend=False,
    )
    return fig
