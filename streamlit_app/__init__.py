"""
Streamlit app package with lazy exports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = ["QualitativeAnalysisApp"]


def __getattr__(name: str) -> Any:
    if name == "QualitativeAnalysisApp":
        return getattr(import_module("streamlit_app.app_core"), name)
    raise AttributeError(f"module 'streamlit_app' has no attribute {name!r}")
