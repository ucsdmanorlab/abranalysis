"""Structural checks on the Plotly figures (not pixel comparisons)."""
import pytest

from utils.plotting import (
    plot_3d_surface,
    plot_waves_single_dB,
    plot_waves_single_frequency,
    plot_waves_single_tuple,
)

pytestmark = pytest.mark.slow

FILE = "20250616 WT.arf"


@pytest.fixture
def loaded(load_demo, apply_settings):
    apply_settings()
    df, _ = load_demo(FILE)
    return df, f"/tmp/{df.name}"


def names(fig):
    return [t.name for t in fig.data]


def test_single_tuple_has_wave_peaks_and_troughs(loaded):
    df, path = loaded
    fig = plot_waves_single_tuple([df], [path], 8000.0, 90.0)
    assert names(fig) == [FILE, "Peaks", "Troughs"]
    assert fig.layout.xaxis.title.text == "Time (ms)"


def test_single_tuple_can_hide_peaks(loaded):
    df, path = loaded
    fig = plot_waves_single_tuple([df], [path], 8000.0, 90.0, show_peaks=False)
    assert names(fig) == [FILE]


def test_single_tuple_unknown_frequency_returns_none(loaded):
    df, path = loaded
    assert plot_waves_single_tuple([df], [path], 123.0, 90.0) is None


def test_single_dB_has_one_wave_per_frequency(loaded):
    df, path = loaded
    figs = plot_waves_single_dB([df], [path], 90.0)
    assert len(figs) == 1
    waves = [n for n in names(figs[0]) if n not in ("Peaks", "Troughs")]
    assert waves == ["8000 Hz", "16000 Hz", "32000 Hz"]


def test_single_dB_level_not_in_file_gives_no_figures(loaded):
    df, path = loaded
    assert plot_waves_single_dB([df], [path], 91.0) == []


def test_single_frequency_returns_figure(loaded):
    df, path = loaded
    figs = plot_waves_single_frequency([df], [path], 8000.0)
    assert len(figs) >= 1 and all(len(f.data) > 0 for f in figs)


def test_units_label_follows_setting(loaded, apply_settings):
    df, path = loaded
    apply_settings(return_units="Nanovolts")
    fig = plot_waves_single_tuple([df], [path], 8000.0, 90.0)
    assert fig.layout.yaxis.title.text == "Voltage (nV)"


def test_fixed_y_range_is_applied(loaded, apply_settings):
    df, path = loaded
    apply_settings(auto_y=False, y_min=-2.0, y_max=2.0)
    fig = plot_waves_single_tuple([df], [path], 8000.0, 90.0)
    assert tuple(fig.layout.yaxis.range) == (-2.0, 2.0)


def test_3d_surface_builds(loaded):
    df, path = loaded
    figs = plot_3d_surface([df], [path], 8000.0)
    assert len(figs) == 1 and len(figs[0].data) >= 1
