"""End-to-end smoke tests that run the real ABRA.py through Streamlit's AppTest.

AppTest cannot upload files and the settings check only runs after an upload, so
these cover startup and the settings-change logic (extracted from ABRA.py); data-processing paths are covered by the other test modules.
"""
import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = [pytest.mark.ui, pytest.mark.slow]


@pytest.fixture(scope="module")
def started():
    return AppTest.from_file("ABRA.py", default_timeout=120).run()


def test_app_starts_without_exceptions(started):
    assert not started.exception


def test_title_and_tabs_present(started):
    assert [t.value for t in started.title] == ["ABRA"]
    assert [t.label for t in started.tabs] == ["Data", "Plotting and Analysis"]


def test_prompts_for_upload_when_no_files(started):
    assert any("upload files" in m.value.lower() for m in started.markdown)


def _load_settings_functions(st_module):
    """ABRA.py runs the whole UI at import, so pull out just the two helper functions."""
    tree = ast.parse(Path("ABRA.py").read_text())
    wanted = {"clear_plots_and_tables", "check_settings_and_clear_cache"}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    namespace = {"st": st_module}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "ABRA.py", "exec"), namespace)
    return namespace


class _FakeSt:
    def __init__(self, session_state):
        self.session_state = session_state
        self.warnings = []

    def warning(self, msg):
        self.warnings.append(msg)

    def expander(self, *a, **k):
        raise AssertionError("expander should only open when manual edits exist")


def test_first_run_records_nothing_and_does_not_warn(session_state):
    fake = _FakeSt(session_state)
    ns = _load_settings_functions(fake)
    ns["check_settings_and_clear_cache"]()
    assert fake.warnings == []


def test_changed_setting_warns_and_clears_calculated_values(session_state):
    fake = _FakeSt(session_state)
    ns = _load_settings_functions(fake)
    session_state["time_scale"] = 5.0
    ns["check_settings_and_clear_cache"]()
    session_state["time_scale"] = 10.0
    session_state["calculated_thresholds"] = {"x": 1}
    session_state["calculated_waves"] = {"y": 2}
    ns["check_settings_and_clear_cache"]()
    assert any("Settings have changed" in w for w in fake.warnings)
    assert not session_state["calculated_thresholds"]
    assert not session_state["calculated_waves"]
    assert session_state["peaks_table"] is None


def test_unchanged_settings_keep_calculated_values(session_state):
    fake = _FakeSt(session_state)
    ns = _load_settings_functions(fake)
    session_state["calculated_thresholds"] = {"x": 1}
    ns["check_settings_and_clear_cache"]()
    ns["check_settings_and_clear_cache"]()
    assert session_state["calculated_thresholds"] == {"x": 1}
    assert fake.warnings == []


def test_attenuation_toggle_with_multiple_files_has_no_duplicate_keys(monkeypatch, tmp_path):
    """Regression: calibration inputs were created once per selected file."""
    import pandas as pd
    from streamlit.delta_generator import DeltaGenerator

    from conftest import UploadedFile

    paths = []
    for i, freq in enumerate((8000, 16000)):
        df = pd.DataFrame(
            [[freq, atten] + [0.0] * 244 for atten in (60, 70, 80)],
            columns=["Freq(Hz)", "PostAtten(dB)"] + [str(n) for n in range(244)],
        )
        paths.append(tmp_path / f"f{i}.csv")
        df.to_csv(paths[-1], index=False)
    monkeypatch.setattr(
        DeltaGenerator, "file_uploader", lambda self, *a, **k: [UploadedFile(p) for p in paths]
    )
    at = AppTest.from_file("ABRA.py", default_timeout=120).run()
    at.toggle(key="atten").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert {n.key for n in at.number_input if str(n.key).startswith("cal_")} == {"cal_8000", "cal_16000"}


def test_tab2_settings_survive_removing_all_files(monkeypatch):
    from streamlit.delta_generator import DeltaGenerator

    from conftest import DEMO_DIR, UploadedFile

    uploaded_paths = [DEMO_DIR / "55.csv"]
    monkeypatch.setattr(
        DeltaGenerator, "file_uploader",
        lambda self, *args, **kwargs: [UploadedFile(path) for path in uploaded_paths],
    )
    at = AppTest.from_file("ABRA.py", default_timeout=120).run()
    at.number_input(key="num_peaks").set_value(3).run()
    at.toggle(key="show_peaks").set_value(False)
    at.selectbox(key="stacked_labels").set_value("Off")
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    uploaded_paths.clear()
    at.run()
    uploaded_paths.append(DEMO_DIR / "80.csv")
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert at.number_input(key="num_peaks").value == 3
    assert at.toggle(key="show_peaks").value is False
    assert at.selectbox(key="stacked_labels").value == "Off"


def test_tables_offer_downloads_with_analysis_settings(monkeypatch):
    import csv
    import io
    import streamlit as st

    from streamlit.delta_generator import DeltaGenerator
    from conftest import DEMO_DIR, UploadedFile

    downloads = {}
    download_button = st.download_button

    def record_download(label, data, **kwargs):
        downloads[kwargs["key"]] = data
        return download_button(label, data, **kwargs)

    monkeypatch.setattr(st, "download_button", record_download)
    monkeypatch.setattr(
        DeltaGenerator, "file_uploader",
        lambda self, *args, **kwargs: [UploadedFile(DEMO_DIR / "20250616 WT.arf")],
    )
    at = AppTest.from_file("ABRA.py", default_timeout=120).run()
    next(button for button in at.button if button.label == "Return all thresholds").click().run()
    assert not at.exception, [e.value for e in at.exception]
    table = at.session_state["threshold_table"]
    assert len(table) > 0
    assert table.attrs["analysis_metadata"]["ABRA code version"]
    assert table.attrs["analysis_metadata"]["Input units"] == "Microvolts"
    assert "ABRA code version" not in table.columns
    assert any(d.proto.label == "Download thresholds (CSV with settings)" for d in at.get("download_button"))
    assert downloads["download_threshold_settings"].startswith(b"\xef\xbb\xbf")
    threshold_rows = list(csv.reader(io.StringIO(downloads["download_threshold_settings"].decode("utf-8-sig"))))
    assert threshold_rows[0][0] == "ABRA code version"
    assert threshold_rows[threshold_rows.index([]) + 1] == list(table.columns) + ["Manual threshold edit"]

    next(button for button in at.button if button.label == "Return all peak analyses").click().run()
    assert not at.exception, [e.value for e in at.exception]
    peak_table = at.dataframe[0].value
    assert len(peak_table) > 0
    assert "Peak I method" not in peak_table.columns
    assert at.session_state["current_table"].attrs["analysis_metadata"]["Peak I method"] == "Pretrained model"
    assert any(d.proto.label == "Download peak analyses (CSV with settings)" for d in at.get("download_button"))
    peak_rows = list(csv.reader(io.StringIO(downloads["download_all_peaks_settings"].decode("utf-8-sig"))))
    assert ["Peak I method", "Pretrained model"] in peak_rows
    assert peak_rows[peak_rows.index([]) + 1] == list(peak_table.columns) + ["Manual peak edit", "Manual threshold used"]


def test_threshold_download_records_visible_editor_change(monkeypatch):
    import csv
    import io
    import streamlit as st

    from streamlit.delta_generator import DeltaGenerator
    from conftest import DEMO_DIR, UploadedFile

    downloads = {}
    download_button = st.download_button

    def record_download(label, data, **kwargs):
        downloads[kwargs["key"]] = data
        return download_button(label, data, **kwargs)

    def edit_threshold(table, **kwargs):
        edited = table.copy()
        edited.loc[edited.index[0], "Estimated Threshold"] = 42.0
        return edited

    monkeypatch.setattr(st, "download_button", record_download)
    monkeypatch.setattr(st, "data_editor", edit_threshold)
    monkeypatch.setattr(
        DeltaGenerator, "file_uploader",
        lambda self, *args, **kwargs: [UploadedFile(DEMO_DIR / "20250616 WT.arf")],
    )
    at = AppTest.from_file("ABRA.py", default_timeout=120).run()
    next(button for button in at.button if button.label == "Return all thresholds").click().run()
    assert not at.exception, [e.value for e in at.exception]
    rows = list(csv.reader(io.StringIO(downloads["download_threshold_settings"].decode("utf-8-sig"))))
    data_rows = rows[rows.index([]) + 1:]
    data = [dict(zip(data_rows[0], row)) for row in data_rows[1:]]
    assert float(data[0]["Estimated Threshold"]) == 42.0
    assert data[0]["Manual threshold edit"] == "True"
    assert data[1]["Manual threshold edit"] == "False"


def test_plot_peak_table_offers_settings_download(monkeypatch):
    import io
    import csv
    import plotly.graph_objects as go
    import streamlit as st
    from streamlit.delta_generator import DeltaGenerator
    from conftest import DEMO_DIR, UploadedFile
    import utils.plotting as plotting

    downloads = {}
    download_button = DeltaGenerator.download_button

    def record_download(self, label, data, **kwargs):
        downloads[kwargs["key"]] = data
        return download_button(self, label, data, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "download_button", record_download)
    monkeypatch.setattr(
        DeltaGenerator, "file_uploader",
        lambda self, *args, **kwargs: [UploadedFile(DEMO_DIR / "20250616 WT.arf")],
    )
    monkeypatch.setattr(plotting, "plot_waves_single_frequency", lambda *args, **kwargs: [go.Figure()])
    monkeypatch.setattr(go.Figure, "write_image", lambda self, file, **kwargs: file.write(b"pdf"))
    at = AppTest.from_file("ABRA.py", default_timeout=120).run()
    next(button for button in at.button if button.label == "Single frequency").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "download_peaks_settings" in downloads
    rows = list(csv.reader(io.StringIO(downloads["download_peaks_settings"].decode("utf-8-sig"))))
    assert ["Peak I method", "Pretrained model"] in rows
    assert "Manual peak edit" in rows[rows.index([]) + 1]

    next(button for button in at.button if button.label == "Edit peaks table").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["editing_peaks_table"]
    assert any("Editing peaks mode" in warning.value for warning in at.warning)
