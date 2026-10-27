"""Tier B: golden baselines on the demo files.

These tests record what ABRA currently reports (thresholds, peak and trough
positions) and fail if a change moves them. Feature work that is not meant to
alter results must leave these untouched. If you change the threshold or
peak-finding algorithm on purpose, run

    pytest tests/test_baselines.py --update-baselines

review `git diff tests/baselines`, and explain the change in the PR (see
tests/README.md).
"""
import pytest

from utils.calculate import calculate_and_plot_wave, calculate_hearing_threshold

pytestmark = [pytest.mark.baseline, pytest.mark.slow]

DEMO_FILES = {
    "wt_arf": "20250616 WT.arf",
    "csv_55": "55.csv",
    "csv_80": "80.csv",
    "tone_baseline_csv": "B1_1282_tone baseline.csv",
}


def _frequencies(df):
    return sorted(df["Freq(Hz)"].unique())


def _levels(df, freq):
    return sorted(df[df["Freq(Hz)"] == freq]["Level(dB)"].unique())


@pytest.mark.parametrize("key", DEMO_FILES)
def test_thresholds_match_baseline(key, load_demo, apply_settings, check):
    apply_settings("default")
    df, _ = load_demo(DEMO_FILES[key])
    thresholds = {str(freq): calculate_hearing_threshold(df, freq) for freq in _frequencies(df)}
    check(f"thresholds_{key}", thresholds)


@pytest.mark.parametrize("preset", ["default", "anchor_pk1", "bk_subtraction"])
@pytest.mark.parametrize("key", DEMO_FILES)
def test_peaks_match_baseline(key, preset, load_demo, apply_settings, check):
    apply_settings(preset)
    df, _ = load_demo(DEMO_FILES[key])
    results = {}
    for freq in _frequencies(df):
        results[str(freq)] = {}
        for db in _levels(df, freq):
            _, _, peaks, troughs = calculate_and_plot_wave(df, freq, db)
            results[str(freq)][str(db)] = {
                "peaks": None if peaks is None else [int(p) for p in peaks],
                "troughs": None if troughs is None else [int(t) for t in troughs],
            }
    check(f"peaks_{key}_{preset}", results)
