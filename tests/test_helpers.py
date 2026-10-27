"""Tests for the helpers and fixtures themselves (fast, no models loaded)."""
import pytest

from baseline_utils import diff_values, to_jsonable, check_baseline


def test_fake_session_state_attribute_and_item_access(session_state):
    import streamlit as st

    st.session_state.foo = 3
    assert st.session_state["foo"] == 3
    assert st.session_state.get("missing", "x") == "x"
    del st.session_state.foo
    assert "foo" not in st.session_state


def test_session_state_has_app_defaults(session_state):
    assert session_state.num_peaks == 5
    assert session_state.units == "Microvolts"


def test_make_abr_shape_and_peaks(make_abr):
    y, peaks, troughs = make_abr()
    assert len(y) == 244
    assert len(peaks) == 5
    assert all(y[p] > 0 for p in peaks)
    assert all(y[t] < 0 for t in troughs)


def test_diff_values_tolerates_small_numeric_noise():
    assert diff_values({"a": [1.0, 2.0]}, {"a": [1.0 + 1e-9, 2.0]}) == []


def test_diff_values_reports_changes_with_path():
    diffs = diff_values({"f": {"t": 40}}, {"f": {"t": 45}})
    assert diffs == ["/f/t: 40 -> 45"]


def test_diff_values_reports_missing_and_new_keys():
    diffs = diff_values({"a": 1}, {"b": 1})
    assert len(diffs) == 2


def test_to_jsonable_converts_numpy_and_nan():
    import numpy as np

    out = to_jsonable({"a": np.array([1, 2]), "b": np.float64("nan"), "c": np.int64(3)})
    assert out == {"a": [1, 2], "b": None, "c": 3}


def test_check_baseline_missing_file_gives_helpful_error(tmp_path, monkeypatch):
    import baseline_utils

    monkeypatch.setattr(baseline_utils, "BASELINE_DIR", tmp_path)
    with pytest.raises(AssertionError, match="update-baselines"):
        check_baseline("nope", {"a": 1}, update=False)


def test_check_baseline_roundtrip_and_mismatch(tmp_path, monkeypatch):
    import baseline_utils

    monkeypatch.setattr(baseline_utils, "BASELINE_DIR", tmp_path)
    check_baseline("x", {"a": 1.0}, update=True)
    check_baseline("x", {"a": 1.0}, update=False)
    with pytest.raises(AssertionError, match="differences"):
        check_baseline("x", {"a": 2.0}, update=False)
