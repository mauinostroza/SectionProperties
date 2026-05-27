"""Session state management for SectionAnalyzer Pro.
Centralizes all st.session_state keys and default values."""

import streamlit as st
import pandas as pd
import numpy as np


def _default_rebar_table() -> pd.DataFrame:
    """Return default reinforcement table with 4 corner bars."""
    return pd.DataFrame({
        "Barra #": [1, 2, 3, 4],
        "Diámetro (mm)": [16.0, 16.0, 16.0, 16.0],
        "X (mm)": [40.0, 260.0, 40.0, 260.0],
        "Y (mm)": [40.0, 40.0, 460.0, 460.0],
        "Capa": ["Ext. inf", "Ext. inf", "Ext. sup", "Ext. sup"],
        "Material": ["B500B", "B500B", "B500B", "B500B"],
    })


def _default_batch_data() -> pd.DataFrame:
    """Return default batch data with 3 sample sections."""
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


def init_session_state() -> None:
    """Initialize all session state keys if not already present."""
    defaults: dict = {
        "active_module": "🔲 Sección",
        "section_type": "Rectangular",
        "b": 300.0,
        "h": 500.0,
        "cover": 40.0,
        "fc": 25.0,
        "fy": 420.0,
        "Ec": 25000.0,
        "Es": 200000.0,
        "units": "kN, m",
        "code": "ACI 318-19",
        "mesh_size": 25.0,
        "project_name": "Edificio A - Pilar P1",
        "rebar_table": _default_rebar_table(),
        "batch_data": _default_batch_data(),
        "batch_results": None,
        "analysis_results": {},
        "section_props": None,
        "settings_open": False,
        "advanced_mesh": False,
        "theme": "Claro",
        "analysis_P": 1200.0,
        "analysis_Mx": 200.0,
        "analysis_My": 50.0,
        "mn_fig": None,
        "mk_fig": None,
        "stress_fig": None,
        "strain_fig": None,
        "cached_section": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_section_params() -> dict:
    """Return dict of current section geometric/material parameters."""
    return {
        "b": st.session_state.b,
        "h": st.session_state.h,
        "cover": st.session_state.cover,
        "fc": st.session_state.fc,
        "fy": st.session_state.fy,
        "Ec": st.session_state.Ec,
        "Es": st.session_state.Es,
    }


def get_rebar_df() -> pd.DataFrame:
    """Return current reinforcement DataFrame."""
    return st.session_state.rebar_table.copy()


def get_units() -> str:
    """Return current unit system string."""
    return st.session_state.units


def get_unit_factor() -> dict:
    """Return conversion factors per current unit system."""
    from config import UNIT_SYSTEMS
    return UNIT_SYSTEMS.get(st.session_state.units, UNIT_SYSTEMS["kN, m"])
