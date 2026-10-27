"""Shared pytest fixtures for the ABRA test suite."""
import io
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo_ABR_files"

# utils/models.py loads model weights with paths relative to the repo root, and
# `utils` must be importable, so make both true no matter where pytest is run.
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore")

from baseline_utils import check_baseline, write_changelog  # noqa: E402

# Settings as the app sets them by default (see ABRA.py).
DEFAULT_SETTINGS = {
    "time_scale": 10.0,
    "multiply_y_factor": 1.0,
    "units": "Microvolts",
    "return_units": "Microvolts",
    "atten": False,
    "peaks_below_thresh": False,
    "num_peaks": 5,
    "pretrained_pk1": True,
    "pk1_anchor": 1.0,
    "max_pk_time": 8.0,
    "pk_bk_subtraction": False,
    "bk_window": 1.0,
    "all_peaks": False,
    "show_legend": True,
    "serif_font": False,
    "auto_y": True,
    "y_min": -5.0,
    "y_max": 5.0,
    "vert_space": 10.0,
}

# Named settings combinations used for golden baselines. Always explicit, so a
# change to an app default never silently changes what a baseline means.
SETTINGS_PRESETS = {
    "default": {},
    "anchor_pk1": {"pretrained_pk1": False, "pk1_anchor": 1.0},
    "bk_subtraction": {"pk_bk_subtraction": True, "bk_window": 1.0},
}


class FakeSessionState(dict):
    """Stand-in for ``st.session_state`` (dict with attribute access)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


class UploadedFile(io.BytesIO):
    """Mimics Streamlit's UploadedFile (a named, readable byte stream)."""

    def __init__(self, path):
        path = Path(path)
        super().__init__(path.read_bytes())
        self.name = path.name


def pytest_addoption(parser):
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Rewrite tests/baselines/*.json from current results (use after an intentional algorithm change).",
    )


_baseline_report = []


def pytest_terminal_summary(terminalreporter, config):
    if not config.getoption("--update-baselines") or not _baseline_report:
        return
    terminalreporter.section("baseline update report")
    for name, kind, diffs in _baseline_report:
        terminalreporter.write_line(f"{name}: {kind}, {len(diffs)} values changed")
        for line in diffs[:15]:
            terminalreporter.write_line(f"    {line}")
        if len(diffs) > 15:
            terminalreporter.write_line(f"    ... and {len(diffs) - 15} more")
    write_changelog(_baseline_report)
    terminalreporter.write_line(
        "Review `git diff tests/baselines`, then fill in the reason in tests/baselines/CHANGELOG.md."
    )


@pytest.fixture
def check(request):
    """``check(name, results)`` compares results to a golden baseline."""
    update = request.config.getoption("--update-baselines")

    def _check(name, results, **kwargs):
        check_baseline(name, results, update, report=_baseline_report, **kwargs)

    return _check


@pytest.fixture
def session_state(monkeypatch):
    """Replace ``st.session_state`` with a fresh dict pre-filled with app defaults."""
    import streamlit as st

    state = FakeSessionState(DEFAULT_SETTINGS)
    state["calibration_levels"] = {}
    monkeypatch.setattr(st, "session_state", state, raising=False)
    return state


@pytest.fixture
def apply_settings(session_state):
    """Apply a named preset (or explicit overrides) on top of the defaults."""

    def _apply(preset="default", **overrides):
        session_state.update(SETTINGS_PRESETS[preset])
        session_state.update(overrides)
        return session_state

    return _apply


@pytest.fixture
def demo_dir():
    return DEMO_DIR


@pytest.fixture
def load_demo(session_state):
    """Load a demo file through the app's own upload pipeline.

    Returns ``(df, duration_ms)``; ``df.name`` is the file name, as in the app.
    """
    from utils.processFiles import process_uploaded_files_cached

    def _load(name, is_rz_file="RZ", click=False, is_atten=False):
        dfs, duration = process_uploaded_files_cached(
            [UploadedFile(DEMO_DIR / name)], is_rz_file, click, is_atten
        )
        return dfs[0], duration

    return _load


@pytest.fixture(scope="session")
def peak_model():
    """The pretrained peak-I CNN (loaded once; slow)."""
    from utils.models import default_peak_finding_model

    return default_peak_finding_model()


@pytest.fixture
def make_abr():
    """Build a synthetic ABR: Gaussian peaks (+) and troughs (-) on a flat baseline.

    Returns ``(waveform, peak_idx, trough_idx)``. Defaults give 5 peaks with
    decreasing amplitude, each followed by a trough, over a 10 ms / 244 sample window.
    """

    def _make(
        n=244,
        window_ms=10.0,
        peak_ms=(1.5, 2.6, 3.7, 4.8, 5.9),
        amplitudes=(1.0, 0.8, 0.9, 0.6, 0.5),
        sigma_ms=0.15,
        noise=0.0,
        seed=0,
        scale=1.0,
    ):
        t = np.linspace(0, window_ms, n)
        y = np.zeros(n)
        peaks = []
        troughs = []
        for ms, amp in zip(peak_ms, amplitudes):
            y += amp * np.exp(-0.5 * ((t - ms) / sigma_ms) ** 2)
            peaks.append(int(np.argmin(np.abs(t - ms))))
            trough_ms = ms + 0.5
            y -= 0.5 * amp * np.exp(-0.5 * ((t - trough_ms) / sigma_ms) ** 2)
            troughs.append(int(np.argmin(np.abs(t - trough_ms))))
        if noise:
            y += np.random.default_rng(seed).normal(0, noise, n)
        return y * scale, np.array(peaks), np.array(troughs)

    return _make
