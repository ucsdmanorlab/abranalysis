"""Session-state logic: db helpers, file selection, manual edits, wave calculation, tables."""
import numpy as np
import pandas as pd
import pytest

from utils.calculate import (
    apply_manual_peak_edits,
    calculate_and_plot_wave,
    calculate_and_plot_wave_exact,
    calculate_hearing_threshold,
    display_peaks_table,
    display_threshold_table,
    get_manual_peak_latency,
)
from utils.processFiles import db_column_name, db_value, get_selected_data
from utils.plotting import apply_units, get_y_units


class TestDbHelpers:
    def test_db_column_name_follows_atten_setting(self, session_state):
        assert db_column_name() == "Level(dB)"
        session_state.atten = True
        assert db_column_name() == "PostAtten(dB)"

    def test_db_value_is_identity_without_attenuation(self, session_state):
        assert db_value("f.arf", 8000.0, 40) == 40

    def test_db_value_subtracts_attenuation_from_calibration(self, session_state):
        session_state.atten = True
        session_state.calibration_levels = {8000.0: 100.0}
        assert db_value("f.arf", 8000.0, "30") == 70.0


class TestUnits:
    def test_microvolts_are_unchanged(self, session_state):
        y = np.array([1.0, -2.0])
        assert np.array_equal(apply_units(y), y)
        assert get_y_units() == "Voltage (μV)"

    def test_nanovolts_scale_by_1000(self, session_state):
        session_state.return_units = "Nanovolts"
        assert np.array_equal(apply_units(np.array([1.0, -2.0])), np.array([1000.0, -2000.0]))
        assert get_y_units() == "Voltage (nV)"


class TestGetSelectedData:
    def _dfs(self, names):
        dfs = []
        for name in names:
            df = pd.DataFrame({"Freq(Hz)": [1.0]})
            df.name = name
            dfs.append(df)
        return dfs

    def test_returns_empty_when_nothing_loaded(self, session_state):
        assert get_selected_data() == ([], [])

    def test_all_files_selected_by_default(self, session_state):
        session_state.processed_dfs = self._dfs(["a.arf", "b.arf"])
        files, dfs = get_selected_data()
        assert [f.split("/")[-1] for f in files] == ["a.arf", "b.arf"]
        assert len(dfs) == 2

    def test_unchecked_files_are_excluded(self, session_state):
        session_state.processed_dfs = self._dfs(["a.arf", "b.arf", "c.arf"])
        session_state["file_1"] = False
        files, dfs = get_selected_data()
        assert [d.name for d in dfs] == ["a.arf", "c.arf"]


class TestManualPeakEdits:
    @pytest.fixture
    def wave(self):
        y = np.zeros(244)
        y[[40, 80, 120]] = [1.0, 0.9, 0.8]
        y[[60, 100]] = [-0.5, -0.4]
        return y

    def test_get_manual_peak_latency(self, session_state):
        assert get_manual_peak_latency("f", 8000.0, 40, "Peak 1 latency (ms)") is None
        session_state.manual_peaks = {"f_8000.0_40": {"Peak 1 latency (ms)": 1.7}}
        assert get_manual_peak_latency("f", 8000.0, 40, "Peak 1 latency (ms)") == 1.7
        assert get_manual_peak_latency("f", 8000.0, 40, "Peak 2 latency (ms)") is None
        assert get_manual_peak_latency("other", 8000.0, 40, "Peak 1 latency (ms)") is None

    def test_no_edits_returns_inputs_unchanged(self, session_state, wave):
        peaks, troughs = np.array([40, 80]), np.array([60, 100])
        out_p, out_t = apply_manual_peak_edits("f", 8000.0, 40, None, wave, peaks, troughs)
        assert out_p is peaks and out_t is troughs

    def test_edit_moves_peak_to_nearby_local_maximum(self, session_state, wave):
        session_state.manual_peaks = {"f_8000.0_40": {"Peak 2 latency (ms)": 120 * 10 / 244}}
        out_p, out_t = apply_manual_peak_edits(
            "f", 8000.0, 40, None, wave, np.array([40, 80]), np.array([60, 100])
        )
        assert list(out_p) == [40, 120]
        assert list(out_t) == [60, 100]

    def test_edit_can_add_a_missing_peak(self, session_state, wave):
        session_state.manual_peaks = {"f_8000.0_40": {"Peak 3 latency (ms)": 120 * 10 / 244}}
        out_p, _ = apply_manual_peak_edits(
            "f", 8000.0, 40, None, wave, np.array([40, 80]), np.array([60, 100])
        )
        assert list(out_p) == [40, 80, 120]

    def test_edit_moves_trough_to_nearby_local_minimum(self, session_state, wave):
        session_state.manual_peaks = {"f_8000.0_40": {"Trough 1 latency (ms)": 100 * 10 / 244}}
        _, out_t = apply_manual_peak_edits(
            "f", 8000.0, 40, None, wave, np.array([40, 80]), np.array([60, 100])
        )
        assert out_t[0] == 100

    def test_nan_edit_is_ignored(self, session_state, wave):
        session_state.manual_peaks = {"f_8000.0_40": {"Peak 1 latency (ms)": np.nan}}
        out_p, _ = apply_manual_peak_edits(
            "f", 8000.0, 40, None, wave, np.array([40, 80]), np.array([60, 100])
        )
        assert list(out_p) == [40, 80]

    def test_edit_outside_waveform_is_ignored(self, session_state, wave):
        session_state.manual_peaks = {"f_8000.0_40": {"Peak 1 latency (ms)": 99.0}}
        out_p, _ = apply_manual_peak_edits(
            "f", 8000.0, 40, None, wave, np.array([40, 80]), np.array([60, 100])
        )
        assert list(out_p) == [40, 80]


@pytest.mark.slow
class TestCalculateAndPlotWave:
    def test_unknown_frequency_returns_nothing(self, load_demo, apply_settings):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        assert calculate_and_plot_wave(df, 12345.0, 40) == (None, None, None, None)

    def test_x_axis_spans_time_scale(self, load_demo, apply_settings):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        x, y, _, _ = calculate_and_plot_wave_exact(df, 8000.0, 80, return_peaks=False)
        assert len(x) == len(y) == 244
        assert x[0] == 0 and x[-1] == pytest.approx(10.0)

    def test_return_peaks_false_skips_peak_finding(self, load_demo, apply_settings):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        _, y, peaks, troughs = calculate_and_plot_wave_exact(df, 8000.0, 80, return_peaks=False)
        assert y is not None and peaks is None and troughs is None

    def test_peaks_found_at_high_level(self, load_demo, apply_settings):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        _, y, peaks, troughs = calculate_and_plot_wave(df, 8000.0, 90)
        assert 1 <= len(peaks) <= 5
        assert np.all(np.diff(peaks) > 0)
        assert peaks.max() < len(y)

    def test_peaks_below_threshold_hidden_unless_enabled(self, load_demo, apply_settings):
        apply_settings(peaks_below_thresh=False)
        df, _ = load_demo("20250616 WT.arf")
        threshold = calculate_hearing_threshold(df, 8000.0)
        below = float(df[(df["Freq(Hz)"] == 8000.0) & (df["Level(dB)"] < threshold)]["Level(dB)"].min())
        _, y, peaks, _ = calculate_and_plot_wave(df, 8000.0, below)
        assert y is not None and peaks is None

    def test_peaks_below_threshold_shown_when_enabled(self, load_demo, apply_settings):
        apply_settings(peaks_below_thresh=True)
        df, _ = load_demo("20250616 WT.arf")
        threshold = calculate_hearing_threshold(df, 8000.0)
        below = float(df[(df["Freq(Hz)"] == 8000.0) & (df["Level(dB)"] < threshold)]["Level(dB)"].min())
        _, _, peaks, _ = calculate_and_plot_wave(df, 8000.0, below)
        assert peaks is not None and len(peaks) > 0

    def test_nanovolt_input_and_multiplier_scale_waveform(self, load_demo, apply_settings):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        _, base, _, _ = calculate_and_plot_wave_exact(df, 8000.0, 80, return_peaks=False)
        from streamlit import session_state

        session_state.units = "Nanovolts"
        session_state.multiply_y_factor = 2.0
        _, scaled_y, _, _ = calculate_and_plot_wave_exact(df, 8000.0, 80, return_peaks=False)
        assert np.allclose(scaled_y, base * 2.0 / 1000)

    def test_results_are_cached(self, load_demo, apply_settings, monkeypatch):
        apply_settings(peaks_below_thresh=True)
        df, _ = load_demo("20250616 WT.arf")
        import utils.calculate as calc

        first = calc.calculate_and_plot_wave(df, 8000.0, 90)
        monkeypatch.setattr(calc, "calculate_and_plot_wave_exact", lambda *a, **k: pytest.fail("cache miss"))
        second = calc.calculate_and_plot_wave(df, 8000.0, 90)
        assert np.array_equal(first[2], second[2])

    def test_manual_peak_edit_overrides_calculated_peak(self, load_demo, apply_settings):
        apply_settings(peaks_below_thresh=True)
        df, _ = load_demo("20250616 WT.arf")
        _, y, peaks, _ = calculate_and_plot_wave(df, 8000.0, 90.0)
        target = int(peaks[1]) + 8
        from streamlit import session_state

        session_state.manual_peaks = {
            f"{df.name}_8000.0_90.0": {"Peak 2 latency (ms)": target * 10.0 / len(y)}
        }
        _, _, edited, _ = calculate_and_plot_wave(df, 8000.0, 90.0)
        assert abs(int(edited[1]) - target) <= 1
        assert edited[0] == peaks[0]


@pytest.mark.slow
class TestTables:
    def test_threshold_table_layout_and_values(self, load_demo, apply_settings):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        freqs = sorted(df["Freq(Hz)"].unique())
        table = display_threshold_table([df], [f"/tmp/{df.name}"], freqs)
        assert list(table.columns) == ["File Name", "Frequency (Hz)", "Estimated Threshold"]
        assert len(table) == len(freqs)
        assert set(table["File Name"]) == {"20250616 WT.arf"}
        for _, row in table.iterrows():
            assert row["Estimated Threshold"] == calculate_hearing_threshold(df, row["Frequency (Hz)"])

    def test_threshold_table_skips_missing_frequencies(self, load_demo, apply_settings):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        table = display_threshold_table([df], [f"/tmp/{df.name}"], [8000.0, 99999.0])
        assert list(table["Frequency (Hz)"]) == [8000.0]

    def test_peaks_table_standard_columns(self, load_demo, apply_settings):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        table = display_peaks_table([df], [f"/tmp/{df.name}"], [8000.0], [60.0, 80.0])
        assert list(table.columns) == [
            "File Name", "Frequency (Hz)", "Sound amplitude (dB SPL)",
            "Wave I amplitude (P1-T1) (μV)", "Latency to First Peak (ms)", "Amplitude Ratio (Peak1/Peak4)",
        ]
        assert list(table["Sound amplitude (dB SPL)"]) == [60.0, 80.0]

    def test_peaks_table_optional_columns(self, load_demo, apply_settings):
        apply_settings(all_peaks=True, num_peaks=3)
        df, _ = load_demo("20250616 WT.arf")
        table = display_peaks_table([df], [f"/tmp/{df.name}"], [8000.0], [80.0], return_threshold=True)
        assert "Estimated Threshold" in table.columns
        for n in (1, 2, 3):
            assert f"Peak {n} latency (ms)" in table.columns
            assert f"Trough {n} (μV)" in table.columns
        assert "Peak 4 latency (ms)" not in table.columns

    def test_editable_table_has_latency_columns_only(self, load_demo, apply_settings):
        apply_settings(num_peaks=4)
        df, _ = load_demo("20250616 WT.arf")
        table = display_peaks_table([df], [f"/tmp/{df.name}"], [8000.0], [80.0], editable=True)
        assert [c for c in table.columns if "latency" in c] == [
            f"{kind} {n} latency (ms)" for n in range(1, 5) for kind in ("Peak", "Trough")
        ]
        assert "Wave I amplitude (P1-T1) (μV)" not in table.columns

    def test_first_peak_latency_matches_peak_1_latency(self, load_demo, apply_settings):
        apply_settings(all_peaks=True)
        df, _ = load_demo("20250616 WT.arf")
        table = display_peaks_table([df], [f"/tmp/{df.name}"], [16000.0], [70.0, 80.0, 90.0])
        assert np.allclose(table["Latency to First Peak (ms)"], table["Peak 1 latency (ms)"], equal_nan=True)

    def test_levels_without_peaks_are_dropped_unless_return_nas(self, load_demo, apply_settings):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        levels = [10.0, 20.0, 90.0]
        dropped = display_peaks_table([df], [f"/tmp/{df.name}"], [8000.0], levels)
        assert list(dropped["Sound amplitude (dB SPL)"]) == [90.0]

        kept = display_peaks_table([df], [f"/tmp/{df.name}"], [8000.0], levels, return_nas=True)
        assert list(kept["Sound amplitude (dB SPL)"]) == levels
        assert pd.isna(kept["Latency to First Peak (ms)"].iloc[0])
        assert not pd.isna(kept["Latency to First Peak (ms)"].iloc[2])

    def test_nanovolt_output_scales_amplitudes(self, load_demo, apply_settings):
        apply_settings()
        df, _ = load_demo("20250616 WT.arf")
        uv = display_peaks_table([df], [f"/tmp/{df.name}"], [8000.0], [90.0])
        from streamlit import session_state

        session_state.return_units = "Nanovolts"
        session_state.calculated_waves.clear()
        nv = display_peaks_table([df], [f"/tmp/{df.name}"], [8000.0], [90.0])
        assert "Wave I amplitude (P1-T1) (nV)" in nv.columns
        assert nv["Wave I amplitude (P1-T1) (nV)"].iloc[0] == pytest.approx(
            uv["Wave I amplitude (P1-T1) (μV)"].iloc[0] * 1000
        )

    @pytest.mark.baseline
    def test_peaks_table_matches_baseline(self, load_demo, apply_settings, check):
        apply_settings(all_peaks=True)
        df, _ = load_demo("20250616 WT.arf")
        table = display_peaks_table(
            [df], [f"/tmp/{df.name}"], [8000.0, 16000.0, 32000.0], [50.0, 70.0, 90.0], return_threshold=True
        )
        check("table_peaks_wt_arf", table.round(4).to_dict(orient="list"), rtol=1e-3)
