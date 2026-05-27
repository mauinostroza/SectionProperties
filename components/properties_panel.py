"""Right-side properties panel wrapper."""

from __future__ import annotations

from collections.abc import Callable
import streamlit as st


def render_properties_panel(title: str, content_func: Callable[[], None]) -> None:
    """Render a titled properties panel with a content function.

    Args:
        title: Panel title (e.g. "Propiedades de la sección")
        content_func: Callable that renders the panel content
    """
    with st.container(border=True):
        st.markdown(f"#### {title}")
        content_func()
