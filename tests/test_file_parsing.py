"""File parsing and the upload pipeline.

The schema tests double as a contract for any new file reader: whatever the
format, the resulting DataFrame must have the same layout so the rest of the
app (thresholds, peaks, plots) keeps working.
"""
import pandas as pd
import pytest

from conftest import DEMO_DIR, UploadedFile
from utils.processFiles import (
    CFTSread,
    arfread,
    is_dpoae_group,
    clear_calculation_cache_for_files,
    process_uploaded_files_cached,
)

DEMO_FILES = ["20250616 WT.arf", "55.csv", "80.csv", "B1_1282_tone baseline.csv"]


def _wave_columns(df):
    return [c for c in df.columns if str(c).isdigit()]


@pytest.mark.parametrize("name", DEMO_FILES)
def test_demo_file_schema_contract(name, load_demo):
    df, _ = load_demo(name)
    assert df.name == name
    assert "Freq(Hz)" in df.columns
    assert "Level(dB)" in df.columns
    wave_cols = _wave_columns(df)
    assert wave_cols[0] == "0"
    assert wave_cols == [str(i) for i in range(len(wave_cols))], "waveform columns must be '0'..'N-1'"
    assert len(wave_cols) > 100
    values = df[wave_cols].apply(pd.to_numeric, errors="coerce")
    assert values.notna().any(axis=1).all(), "every row must contain waveform data"
    assert not df[["Freq(Hz)", "Level(dB)"]].isna().any().any()


def test_arf_demo_contents(load_demo):
    df, duration = load_demo("20250616 WT.arf")
    assert sorted(df["Freq(Hz)"].unique()) == [8000.0, 16000.0, 32000.0]
    assert df["Level(dB)"].min() == 10.0
    assert df["Level(dB)"].max() == 90.0
    assert len(_wave_columns(df)) == 244
    assert duration == pytest.approx(10.0, abs=0.01)


def test_arf_values_are_converted_to_microvolts(load_demo):
    df, _ = load_demo("20250616 WT.arf")
    peak_abs = df[_wave_columns(df)].abs().to_numpy().max()
    # Raw ARF samples are volts (~1e-6); the pipeline scales by 1e6.
    assert 0.01 < peak_abs < 1e4


def test_arf_click_mode_uses_click_label(load_demo):
    df, _ = load_demo("20250616 WT.arf", click=True)
    assert (df["Freq(Hz)"] == "Click").all()


def test_arf_attenuation_mode_uses_postatten_column(load_demo):
    df, _ = load_demo("20250616 WT.arf", is_atten=True)
    assert "PostAtten(dB)" in df.columns
    assert "Level(dB)" not in df.columns


def test_arfread_structure(demo_dir):
    data = arfread(str(demo_dir / "20250616 WT.arf"), RP=False)
    assert data["fileType"] == "BioSigRZ"
    assert len(data["groups"]) == data["metadata"]["ngrps"]
    rec = data["groups"][0]["recs"][0]
    assert rec["npts"] == len(rec["data"])


def test_arfread_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.arf"
    bad.write_bytes(b"\x00\x01")
    with pytest.raises(Exception):
        arfread(str(bad))


def test_csv_without_freq_column_gets_zero_freq(tmp_path, session_state):
    path = tmp_path / "nofreq.csv"
    pd.DataFrame({"Level(dB)": [20, 40], "0": [0.1, 0.2], "1": [0.3, 0.4]}).to_csv(path, index=False)
    dfs, _ = process_uploaded_files_cached([UploadedFile(path)], "RZ", False, False)
    df = dfs[0]
    assert list(df.columns)[:2] == ["Freq(Hz)", "Level(dB)"]
    assert (df["Freq(Hz)"] == 0.0).all()


def _write_tsv(path, freq_line=":FREQ: 8.0", levels=(20, 40, 60), n=50, sample_us=40):
    lines = [
        ":RUN-1",
        freq_line,
        ":LEVELS:" + ";".join(str(level) for level in levels) + ";",
        f":SAMPLE (us): {sample_us}",
        ":DATA",
    ]
    for i in range(n):
        lines.append(" ".join(str(i * 0.01 * (k + 1)) for k in range(len(levels))))
        lines.append("")
    path.write_text("\n".join(lines))


def test_cftsread_parses_tone_file(tmp_path):
    path = tmp_path / "run.tsv"
    _write_tsv(path)
    duration, df = CFTSread(str(path))
    assert duration == pytest.approx(49 * 40 / 1000)
    assert list(df["Level(dB)"]) == [20, 40, 60]
    assert (df["Freq(Hz)"] == 8000.0).all()
    assert len(_wave_columns(df)) == 50
    assert df.loc[1, "1"] == pytest.approx(0.02)


def test_cftsread_click_file(tmp_path):
    path = tmp_path / "click.tsv"
    _write_tsv(path, freq_line=":FREQ: Click")
    _, df = CFTSread(str(path))
    assert (df["Freq(Hz)"] == "Click").all()


def test_upload_pipeline_reads_tsv(tmp_path, session_state):
    path = tmp_path / "run.tsv"
    _write_tsv(path)
    dfs, duration = process_uploaded_files_cached([UploadedFile(path)], "RZ", False, False)
    assert dfs[0].name == "run.tsv"
    assert duration == pytest.approx(49 * 40 / 1000)


def test_upload_pipeline_caches_identical_requests(session_state):
    files = [UploadedFile(DEMO_DIR / "55.csv")]
    first, _ = process_uploaded_files_cached(files, "RZ", False, False)
    second, _ = process_uploaded_files_cached([UploadedFile(DEMO_DIR / "55.csv")], "RZ", False, False)
    assert first is second


def test_upload_pipeline_rereads_when_settings_change(session_state):
    first, _ = process_uploaded_files_cached([UploadedFile(DEMO_DIR / "20250616 WT.arf")], "RZ", False, False)
    second, _ = process_uploaded_files_cached([UploadedFile(DEMO_DIR / "20250616 WT.arf")], "RZ", False, True)
    assert first is not second
    assert "PostAtten(dB)" in second[0].columns


def test_upload_pipeline_handles_multiple_files(session_state):
    files = [UploadedFile(DEMO_DIR / "55.csv"), UploadedFile(DEMO_DIR / "80.csv")]
    dfs, _ = process_uploaded_files_cached(files, "RZ", False, False)
    assert [d.name for d in dfs] == ["55.csv", "80.csv"]


def test_upload_pipeline_unsupported_extension_gives_clear_error(tmp_path, session_state):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(ValueError):
        process_uploaded_files_cached([UploadedFile(path)], "RZ", False, False)


def test_clear_calculation_cache_only_removes_matching_files(session_state):
    session_state.calculated_thresholds = {"threshold_a.arf_8000": 1, "threshold_b.arf_8000": 2}
    session_state.calculated_waves = {"wave_a.arf_8000_20": 1, "wave_b.arf_8000_20": 2}
    clear_calculation_cache_for_files(["a.arf"])
    assert session_state.calculated_thresholds == {"threshold_b.arf_8000": 2}
    assert session_state.calculated_waves == {"wave_b.arf_8000_20": 2}


def test_clear_calculation_cache_without_caches_is_a_noop(session_state):
    clear_calculation_cache_for_files(["a.arf"])


def _group(protocol, var1, var2, recs):
    return {"sgfname1": protocol, "sgfname2": "", "VarName1": var1, "VarName2": var2, "recs": recs}


def _rec(var1, var2, n, dur):
    return {"Var1": var1, "Var2": var2, "data": [1e-6] * n, "dur_ms": dur}


def test_is_dpoae_group_by_protocol_name_or_variables():
    assert is_dpoae_group(_group("DPOAE1A_IO_8", "Level", "AudFreq", []))
    assert is_dpoae_group(_group("custom", "Level", "AudFreq", []))
    assert not is_dpoae_group(_group("zCS_LC TonePip upd", "Freq", "Level", []))
    assert not is_dpoae_group(_group("", "Level", "", []))


def test_upload_ignores_dpoae_groups_in_arf(tmp_path, session_state, monkeypatch):
    groups = [
        _group("DPOAE1A_IO_8", "Level", "AudFreq", [_rec(30.0, 8000.0, 2048, 97656.25)]),
        _group("zCS_LC TonePip", "Freq", "Level", [_rec(8000.0, 20.0, 244, 10.0), _rec(8000.0, 30.0, 244, 10.0)]),
    ]
    monkeypatch.setattr("utils.processFiles.arfread", lambda *a, **k: {"groups": groups})
    path = tmp_path / "mixed.arf"
    path.write_bytes(b"x")
    dfs, durations = process_uploaded_files_cached([UploadedFile(path)], "RZ", False, False)
    df = dfs[0]
    assert list(df["Freq(Hz)"]) == [8000.0, 8000.0]
    assert list(df["Level(dB)"]) == [20.0, 30.0]
    assert df.shape[1] == 2 + 244
    assert durations == 10.0


def test_upload_arf_with_only_dpoae_gives_clear_error(tmp_path, session_state, monkeypatch):
    groups = [_group("DPOAE1A_IO_8", "Level", "AudFreq", [_rec(30.0, 8000.0, 2048, 97656.25)])]
    monkeypatch.setattr("utils.processFiles.arfread", lambda *a, **k: {"groups": groups})
    path = tmp_path / "dpoae_only.arf"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="No ABR"):
        process_uploaded_files_cached([UploadedFile(path)], "RZ", False, False)
