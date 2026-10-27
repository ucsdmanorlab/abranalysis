"""Tier A: contracts for peak finding and thresholding.

These describe what any correct algorithm must satisfy (ordering, ranges,
counts, agreement with synthetic ground truth). They should keep passing when
the algorithm is improved; only change them if the contract itself changes.
"""
import contextlib
import signal
import sys

import numpy as np
import pytest

from utils.calculate import (
    compute_hearing_threshold,
    interpolate_and_smooth,
    peak_finding,
)

NO_MODEL_KWARGS = dict(pretrained_pk1=False, pk1_anchor=1.5, max_pk_time=10.0, bk_window=1.0)


def scaled(y):
    """Min-max scale like the app does before peak finding."""
    return (y - y.min()) / (y.max() - y.min())


@contextlib.contextmanager
def time_limit(seconds):
    if sys.platform == "win32":
        pytest.skip("signal-based time limit needs a POSIX platform")

    def handler(signum, frame):
        raise TimeoutError(f"still running after {seconds}s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


class TestPeakFindingSynthetic:
    def test_recovers_known_peaks(self, make_abr):
        y, true_peaks, _ = make_abr()
        peaks, _ = peak_finding(scaled(y), None, n_peaks=5, **NO_MODEL_KWARGS)
        assert len(peaks) == 5
        assert np.all(np.abs(peaks - true_peaks) <= 2)

    def test_recovers_known_troughs_after_each_peak(self, make_abr):
        y, true_peaks, true_troughs = make_abr()
        _, troughs = peak_finding(scaled(y), None, n_peaks=5, **NO_MODEL_KWARGS)
        assert len(troughs) == 5
        assert np.all(np.abs(troughs - true_troughs) <= 3)

    def test_robust_to_small_noise(self, make_abr):
        y, true_peaks, _ = make_abr(noise=0.02, seed=1)
        peaks, _ = peak_finding(scaled(y), None, n_peaks=5, **NO_MODEL_KWARGS)
        assert len(peaks) == 5
        assert np.all(np.abs(peaks - true_peaks) <= 3)

    def test_invariant_to_amplitude_and_offset(self, make_abr):
        y, _, _ = make_abr()
        base, _ = peak_finding(scaled(y), None, n_peaks=5, **NO_MODEL_KWARGS)
        other, _ = peak_finding(scaled(y * 7.5 + 3.0), None, n_peaks=5, **NO_MODEL_KWARGS)
        assert np.array_equal(base, other)

    @pytest.mark.parametrize("n_peaks", [
        1, 2, 3, 4, 5,
    ])
    def test_never_returns_more_than_requested_peaks(self, make_abr, n_peaks):
        y, _, _ = make_abr()
        peaks, troughs = peak_finding(scaled(y), None, n_peaks=n_peaks, **NO_MODEL_KWARGS)
        assert len(peaks) == n_peaks
        assert len(troughs) <= len(peaks)

    def test_first_peak_is_always_kept(self, make_abr):
        y, true_peaks, _ = make_abr()
        peaks, _ = peak_finding(scaled(y), None, n_peaks=2, **NO_MODEL_KWARGS)
        assert abs(peaks[0] - true_peaks[0]) <= 2

    def test_peaks_sorted_in_time_and_inside_window(self, make_abr):
        y, _, _ = make_abr(noise=0.05, seed=3)
        peaks, troughs = peak_finding(scaled(y), None, n_peaks=5, **NO_MODEL_KWARGS)
        assert np.all(np.diff(peaks) > 0)
        assert np.all(np.diff(troughs) > 0)
        assert peaks.min() >= 0 and peaks.max() < 244

    def test_each_trough_lies_between_its_peak_and_the_next(self, make_abr):
        y, _, _ = make_abr(noise=0.02, seed=2)
        peaks, troughs = peak_finding(scaled(y), None, n_peaks=5, **NO_MODEL_KWARGS)
        for i, trough in enumerate(troughs):
            assert trough > peaks[i]
            if i + 1 < len(peaks):
                assert trough < peaks[i + 1]

    def test_max_pk_time_excludes_late_peaks(self, make_abr):
        y, true_peaks, _ = make_abr()
        peaks, _ = peak_finding(
            scaled(y), None, n_peaks=5, **{**NO_MODEL_KWARGS, "max_pk_time": 3.0}
        )
        assert peaks.max() < 3.0 / 10 * 244
        assert len(peaks) == 2

    def test_pk1_anchor_controls_which_peak_is_first(self, make_abr):
        y, true_peaks, _ = make_abr()
        peaks, _ = peak_finding(
            scaled(y), None, n_peaks=5, **{**NO_MODEL_KWARGS, "pk1_anchor": 3.7}
        )
        assert abs(peaks[0] - true_peaks[2]) <= 2
        assert len(peaks) == 3

    def test_background_subtraction_still_finds_peaks_on_drifting_baseline(self, make_abr):
        y, true_peaks, _ = make_abr()
        drift = np.linspace(0, 1.5, 244)
        peaks, _ = peak_finding(
            scaled(y + drift), None, running_avg_method=True, n_peaks=5, **NO_MODEL_KWARGS
        )
        assert len(peaks) >= 3
        assert abs(peaks[0] - true_peaks[0]) <= 3

    def test_reads_settings_from_session_when_not_given(self, make_abr, apply_settings):
        y, true_peaks, _ = make_abr()
        apply_settings(pretrained_pk1=False, pk1_anchor=1.5, num_peaks=3, max_pk_time=10.0)
        peaks, _ = peak_finding(scaled(y), None)
        assert len(peaks) == 3

    def test_explicit_settings_override_session(self, make_abr, apply_settings):
        y, _, _ = make_abr()
        apply_settings(pretrained_pk1=False, pk1_anchor=1.5, num_peaks=2, max_pk_time=10.0)
        peaks, _ = peak_finding(scaled(y), None, n_peaks=4)
        assert len(peaks) == 4

    def test_waveform_without_peaks_returns_nothing_instead_of_hanging(self):
        with time_limit(5):
            peaks, troughs = peak_finding(np.zeros(244), None, n_peaks=5, **NO_MODEL_KWARGS)
        assert len(peaks) == 0 and len(troughs) == 0


@pytest.mark.slow
class TestPeakFindingWithPretrainedModel:
    def test_model_prediction_is_a_plausible_sample_index(self, peak_model, make_abr):
        import torch

        y, _, _ = make_abr()
        out = peak_model(torch.tensor(scaled(y), dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        assert out.shape == (1, 1)
        assert -50 < float(out.item()) < 300

    def test_peaks_and_troughs_satisfy_structural_contracts(self, peak_model, make_abr):
        y, _, _ = make_abr(noise=0.02, seed=5)
        peaks, troughs = peak_finding(
            scaled(y), peak_model, n_peaks=5, pretrained_pk1=True, pk1_anchor=1.0, max_pk_time=8.0, bk_window=1.0
        )
        assert 1 <= len(peaks) <= 5
        assert np.all(np.diff(peaks) > 0)
        assert peaks.max() < 8.0 / 10 * 244
        assert len(troughs) <= len(peaks)

    def test_deterministic(self, peak_model, make_abr):
        y, _, _ = make_abr(noise=0.02, seed=5)
        kwargs = dict(n_peaks=5, pretrained_pk1=True, pk1_anchor=1.0, max_pk_time=8.0, bk_window=1.0)
        a, _ = peak_finding(scaled(y), peak_model, **kwargs)
        b, _ = peak_finding(scaled(y), peak_model, **kwargs)
        assert np.array_equal(a, b)


class TestInterpolateAndSmooth:
    @pytest.mark.parametrize("length", [100, 244, 500, 1000])
    def test_output_has_target_length(self, length):
        out = interpolate_and_smooth(np.sin(np.linspace(0, 6, length)), 244)
        assert len(out) == 244

    @pytest.mark.parametrize("length", [100, 500])
    def test_endpoints_preserved(self, length):
        y = np.linspace(2.0, 9.0, length)
        out = interpolate_and_smooth(y, 244)
        assert out.iloc[0] == pytest.approx(2.0)
        assert out.iloc[-1] == pytest.approx(9.0)

    def test_same_length_is_unchanged(self):
        y = np.random.default_rng(0).normal(size=244)
        assert np.allclose(interpolate_and_smooth(y, 244).to_numpy(), y)

    def test_linear_signal_stays_linear(self):
        out = interpolate_and_smooth(np.linspace(0, 1, 488), 244)
        assert np.allclose(out.to_numpy(), np.linspace(0, 1, 244))


@pytest.mark.slow
class TestThreshold:
    """Thresholds use the real Keras model, so these check behavior, not exact values."""

    def test_threshold_is_one_of_the_measured_levels(self, load_demo, apply_settings):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        from utils.calculate import calculate_hearing_threshold

        for freq in sorted(df["Freq(Hz)"].unique()):
            levels = set(df[df["Freq(Hz)"] == freq]["Level(dB)"])
            assert calculate_hearing_threshold(df, freq) in levels

    def test_threshold_is_cached_in_session(self, load_demo, apply_settings, monkeypatch):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        import utils.calculate as calc

        first = calc.calculate_hearing_threshold(df, 8000.0)
        monkeypatch.setattr(calc, "compute_hearing_threshold", lambda *a, **k: pytest.fail("should be cached"))
        assert calc.calculate_hearing_threshold(df, 8000.0) == first

    def test_manual_threshold_overrides_calculation(self, load_demo, apply_settings):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        from utils.calculate import calculate_hearing_threshold

        calculated = calculate_hearing_threshold(df, 8000.0)
        manual = 85.0 if calculated != 85.0 else 80.0
        from streamlit import session_state

        session_state.manual_thresholds = {f"{df.name}_8000.0": manual}
        assert calculate_hearing_threshold(df, 8000.0) == manual

    def test_explicit_settings_match_session_path(self, load_demo, apply_settings):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        from utils.calculate import calculate_hearing_threshold

        explicit = compute_hearing_threshold(
            df, 16000.0, atten=False, time_scale=10.0, multiply_y_factor=1.0, units="Microvolts"
        )
        assert explicit == calculate_hearing_threshold(df, 16000.0)

    def test_attenuation_mode_gives_same_threshold_as_spl_mode(self, load_demo, apply_settings):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        atten_df = df.rename(columns={"Level(dB)": "PostAtten(dB)"}).copy()
        atten_df["PostAtten(dB)"] = 100.0 - atten_df["PostAtten(dB)"]
        expected = compute_hearing_threshold(
            df, 16000.0, atten=False, time_scale=10.0, multiply_y_factor=1.0, units="Microvolts"
        )
        actual = compute_hearing_threshold(
            atten_df, 16000.0, atten=True, time_scale=10.0, multiply_y_factor=1.0, units="Microvolts",
            calibration_level=100.0,
        )
        assert actual == expected

    def test_amplitude_unit_conversion_does_not_change_threshold(self, load_demo, apply_settings):
        # Waveforms are min-max scaled before the model, so a pure rescale must not matter.
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        base = compute_hearing_threshold(
            df, 8000.0, atten=False, time_scale=10.0, multiply_y_factor=1.0, units="Microvolts"
        )
        scaled_result = compute_hearing_threshold(
            df, 8000.0, atten=False, time_scale=10.0, multiply_y_factor=3.0, units="Microvolts"
        )
        assert scaled_result == base
