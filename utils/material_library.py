"""Material library: creates sectionproperties Material objects from presets."""

from __future__ import annotations

from sectionproperties.pre.pre import Material
from config import MATERIAL_PRESETS


def get_concrete_material(
    fc: float,
    Ec: float = 25000.0,
    color: str = "lightgrey",
    name: str = "Concrete",
) -> Material:
    """Create a concrete Material for sectionproperties.

    Args:
        fc: Compressive strength in MPa
        Ec: Elastic modulus in MPa
        color: Display color
        name: Material name

    Returns:
        Material object
    """
    return Material(
        name=name,
        elastic_modulus=Ec,
        poissons_ratio=0.2,
        yield_strength=fc,
        density=2.4e-6,  # kg/mm³ (24 kN/m³)
        color=color,
    )


def get_steel_material(
    fy: float,
    Es: float = 200000.0,
    color: str = "grey",
    name: str = "Steel",
) -> Material:
    """Create a steel Material for sectionproperties.

    Args:
        fy: Yield strength in MPa
        Es: Elastic modulus in MPa
        color: Display color
        name: Material name

    Returns:
        Material object
    """
    return Material(
        name=name,
        elastic_modulus=Es,
        poissons_ratio=0.3,
        yield_strength=fy,
        density=7.85e-6,  # kg/mm³ (78.5 kN/m³)
        color=color,
    )


def get_material_presets(code: str = "ACI 318-19") -> dict:
    """Return material preset dict for a given design code."""
    return MATERIAL_PRESETS


def get_concrete_preset(name: str = "H-25") -> dict:
    """Get a concrete preset by name."""
    return MATERIAL_PRESETS.get("Concrete", {}).get(name, MATERIAL_PRESETS["Concrete"]["H-25"])


def get_steel_preset(name: str = "B500B") -> dict:
    """Get a steel preset by name."""
    return MATERIAL_PRESETS.get("Steel", {}).get(name, MATERIAL_PRESETS["Steel"]["B500B"])
