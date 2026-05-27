"""Geometry factory: creates sectionproperties Geometry objects from parameter dicts."""

from __future__ import annotations
from typing import Any, Optional
import pandas as pd

from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.library import (
    rectangular_section,
    circular_section,
    tee_section,
    i_section,
    channel_section,
    angle_section,
    rectangular_hollow_section,
)


def create_section(section_type: str, params: dict[str, Any], material=None) -> Geometry | None:
    """Create a sectionproperties Geometry from type and params.

    Args:
        section_type: One of 'Rectangular', 'Circular', 'T', 'I', 'L', 'Cajón'
        params: Dict with geometric parameters
        material: Optional Material object

    Returns:
        Geometry object or None if type not recognized.
    """
    factory = {
        "Rectangular": _create_rectangular,
        "Circular": _create_circular,
        "T": _create_T,
        "I": _create_I,
        "L": _create_L,
        "Cajón": _create_box,
    }
    creator = factory.get(section_type)
    if creator is None:
        return None
    return creator(params, material)


def _create_rectangular(params: dict[str, Any], material) -> Geometry:
    b = params.get("b", 300.0)
    h = params.get("h", 500.0)
    return rectangular_section(d=h, b=b, material=material)


def _create_circular(params: dict[str, Any], material) -> Geometry:
    d = params.get("d", 500.0)
    return circular_section(d=d, n=32, material=material)


def _create_T(params: dict[str, Any], material) -> Geometry:
    b_f = params.get("b_f", 800.0)
    h_f = params.get("h_f", 150.0)
    b_w = params.get("b_w", 300.0)
    h_w = params.get("h_w", 450.0)
    return tee_section(
        b=b_f, d=h_f + h_w, t_f=h_f, t_w=b_w, r=0, n_r=4,
        material=material,
    )


def _create_I(params: dict[str, Any], material) -> Geometry:
    b_f = params.get("b_f", 300.0)
    h_f = params.get("h_f", 500.0)
    b_w = params.get("b_w", 150.0)
    h_w = params.get("h_w", 500.0)
    t_f = params.get("t_f", 20.0)
    return i_section(
        d=h_f, b_f=b_f, t_f=t_f, t_w=b_w, r=0, n_r=4,
        material=material,
    )


def _create_L(params: dict[str, Any], material) -> Geometry:
    b = params.get("b", 150.0)
    h = params.get("h", 150.0)
    t = params.get("t", 10.0)
    return angle_section(
        d=h, b=b, t=t, r=0, n_r=4,
        material=material,
    )


def _create_box(params: dict[str, Any], material) -> Geometry:
    b = params.get("b", 600.0)
    d = params.get("d", 400.0)
    t = params.get("t", 30.0)
    return rectangular_hollow_section(
        b=b, d=d, t=t, n_r=4, material=material,
    )


def concrete_section_from_params(
    geom_params: dict,
    concrete_material: Any = None,
    steel_material: Any = None,
    rebar_df: Optional[pd.DataFrame] = None,
) -> Geometry | None:
    """Create a reinforced concrete rectangular section geometry.
    Falls back to plain rectangular section if no rebar data.
    """
    import numpy as np
    b = geom_params.get("b", 300.0)
    h = geom_params.get("h", 500.0)

    if concrete_material is None:
        from utils.material_library import get_concrete_material
        concrete_material = get_concrete_material(geom_params.get("fc", 25))

    if rebar_df is not None and len(rebar_df) > 0:
        try:
            from sectionproperties.pre.library.concrete_sections import concrete_rectangular_section
            # Rebar geometry embedded in concrete section
            return concrete_rectangular_section(
                b=b, d=h, cover=geom_params.get("cover", 40.0),
                area_concrete=concrete_material,
                area_steel=steel_material,
                n_x=4, n_y=4,
                n_cx=4, n_cy=4,
            )
        except (ImportError, TypeError):
            pass

    return rectangular_section(d=h, b=b, material=concrete_material)
