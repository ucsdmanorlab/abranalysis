import io
import csv
import subprocess

import pandas as pd
import pytest

from utils import analysis_metadata as metadata


@pytest.mark.parametrize(
    ("settings", "method", "anchor", "background", "window"),
    [
        ({"pretrained_pk1": True, "pk1_anchor": 2.0, "pk_bk_subtraction": False, "bk_window": 2.5},
         "Pretrained model", "", False, ""),
        ({"pretrained_pk1": False, "pk1_anchor": 1.7, "pk_bk_subtraction": True, "bk_window": 2.5},
         "Search around time", 1.7, True, 2.5),
    ],
)
def test_peak_settings_are_recorded_in_csv(monkeypatch, settings, method, anchor, background, window):
    monkeypatch.setattr(metadata, "code_version", lambda: "v1.0-2-gabcdef")
    settings.update(num_peaks=3, max_pk_time=7.5, peaks_below_thresh=True,
                    time_scale=12.0, units="Nanovolts", multiply_y_factor=2.0, atten=True, all_peaks=True)
    table = metadata.add_analysis_metadata(pd.DataFrame({"value": [1, 2]}), settings, peaks=True)
    assert list(table.columns) == ["value"]
    snapshot = table.attrs["analysis_metadata"]
    output = metadata.table_with_settings_csv(table, snapshot)
    assert output.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(output.decode("utf-8-sig"))))
    separator = rows.index([])
    header = dict(rows[:separator])
    assert header["ABRA code version"] == "v1.0-2-gabcdef"
    assert header["Peak I method"] == method
    assert header["Peak I search time (ms)"] == str(anchor)
    assert header["Background subtraction"] == str(background)
    assert header["Background window (ms)"] == str(window)
    assert header["Number of peaks"] == "3"
    assert header["Peak cutoff time (ms)"] == "7.5"
    assert header["Peaks below threshold"] == "True"
    assert header["Input time scale (ms)"] == "12.0"
    assert header["Input units"] == "Nanovolts"
    assert header["Y multiplier"] == "2.0"
    assert header["Input levels are attenuation"] == "True"
    assert header["All peaks in table"] == "True"
    assert rows[separator + 1:] == [["value"], ["1"], ["2"]]


def test_threshold_table_gets_common_settings_but_not_peak_settings(monkeypatch):
    monkeypatch.setattr(metadata, "code_version", lambda: "v1.0-2-gabcdef")
    table = metadata.add_analysis_metadata(pd.DataFrame({"value": [1]}), {}, peaks=False)
    saved = table.attrs["analysis_metadata"]
    assert saved["ABRA code version"] == "v1.0-2-gabcdef"
    assert saved["Input time scale (ms)"] == 10.0
    assert "Peak I method" not in saved
    assert list(table.columns) == ["value"]


def test_csv_escapes_values_and_preserves_empty_table():
    table = pd.DataFrame(columns=["File Name", "value"])
    output = metadata.table_with_settings_csv(table, {"setting": 'name, "quoted"'})
    rows = list(csv.reader(io.StringIO(output.decode("utf-8-sig"))))
    assert rows[0] == ["setting", 'name, "quoted"']
    assert rows[-1] == ["File Name", "value"]


def test_csv_preserves_microvolts_and_manual_flags():
    table = pd.DataFrame({"Wave I amplitude (P1-T1) (μV)": [1, 2]}, index=[3, 5])
    output = metadata.table_with_settings_csv(
        table, {}, {"Manual peak edit": {3: True, 5: False}}
    )
    assert output.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(output.decode("utf-8-sig"))))
    assert rows[-3:] == [
        ["Wave I amplitude (P1-T1) (μV)", "Manual peak edit"],
        ["1", "True"], ["2", "False"],
    ]


def test_version_failure_is_explicit(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("git", 5)

    monkeypatch.setattr(metadata.subprocess, "run", fail)
    assert metadata.code_version() == "Unavailable (TimeoutExpired)"


def test_version_marks_tracked_edits_in_a_git_checkout(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "utils").mkdir()
    monkeypatch.setattr(metadata, "__file__", str(root / "utils" / "analysis_metadata.py"))
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    tracked = root / "file.txt"
    tracked.write_text("before\n")
    subprocess.run(["git", "-C", str(root), "add", "file.txt"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "test"], check=True)
    clean = metadata.code_version()
    assert not clean.endswith("-dirty")
    tracked.write_text("after\n")
    assert metadata.code_version() == clean + "-dirty"
