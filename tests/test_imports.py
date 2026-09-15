"""Smoke tests that verify ABRA's dependencies and internal modules import cleanly.

These tests intentionally do NOT run the Streamlit app itself (ABRA.py executes
Streamlit page-config calls at module scope, which requires a live Streamlit
runtime). Instead, they exercise the real dependency surface: the third-party
packages listed in requirements.txt and every module in utils/, which is where
platform-specific install/import issues (torch, tensorflow, scikit-fda,
fdasrsf, etc.) tend to show up.

Run with: pytest tests/test_imports.py -v
"""
import importlib

import pytest

THIRD_PARTY_MODULES = [
    "streamlit",
    "fdasrsf",
    "plotly",
    "pandas",
    "numpy",
    "scipy",
    "backports.tempfile",
    "skfda",
    "kneed",
    "torch",
    "keras",
    "tensorflow",
    "matplotlib",
    "colorcet",
    "kaleido",
    "sklearn",
]

UTILS_MODULES = [
    "utils.ui",
    "utils.processFiles",
    "utils.models",
    "utils.calculate",
    "utils.plotting",
    "utils.legacy",
]


@pytest.mark.parametrize("module_name", THIRD_PARTY_MODULES)
def test_third_party_import(module_name):
    """Each third-party dependency should import without raising."""
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", UTILS_MODULES)
def test_utils_module_import(module_name):
    """Each ABRA utils submodule should import without raising."""
    importlib.import_module(module_name)
