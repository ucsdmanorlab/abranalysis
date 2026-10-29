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
    awfread,
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


def _write_awf(path, recs, rz=True):
    """Build a minimal .awf: header, 30 record headers, then the waveforms."""
    import struct
    import time

    def s(text, n):
        return text.encode().ljust(n, b"\x00")

    blob = struct.pack("<hffhhhhhhhhh", len(recs), 1e-6, -1e-6, 1, 10, 0, 1, 1, 0, 0, 0, 0)
    ts = int(time.time()) - 1000
    for i in range(30):
        rec = recs[i] if i < len(recs) else None
        t = struct.pack("<q", ts) if rz else struct.pack("<i", ts)
        blob += struct.pack("<hh", 1, 0) + t + struct.pack("<hhbbh", 1, 1, 1, 0, len(rec["data"]) if rec else 0)
        blob += struct.pack("<fff", 0.0, rec["dur"] if rec else 0.0, 0.0) + struct.pack("<ffhhh", 0, 0, 0, 0, 0)
        blob += (struct.pack("<qq", ts, ts) if rz else struct.pack("<ii", ts, ts))
        blob += struct.pack("<10f", *(rec["vars"] if rec else [0.0] * 10))
        blob += b"\x00" * (10 * 36)
        blob += struct.pack("<hhh", 1, 0, 1) + s("ID1", 16) + s("", 32) + s("memo", 50)
        blob += (struct.pack("<qq", ts, ts) if rz else struct.pack("<ii", ts, ts))
        blob += s("proto", 100) + s("", 100)
        blob += b"".join(s(n, 15) for n in ["Freq", "Level"] + [""] * 8) + b"".join(s("", 5) for _ in range(10))
        blob += struct.pack("<f", 40.0) + b"\x00" * (4 + 2 + 4 + 92)
        blob += struct.pack("<h", (20 + i) if rec else 0) + b"\x00" * (18 + 5 * 24 + 108)
    for rec in recs:
        blob += struct.pack(f"<{len(rec['data'])}f", *rec["data"])
    path.write_bytes(blob)


@pytest.mark.parametrize("rz", [True, False])
def test_awfread_roundtrip(tmp_path, rz):
    recs = [
        {"vars": [8000.0, 90.0] + [0.0] * 8, "dur": 10.0, "data": [1e-6, 2e-6, 3e-6]},
        {"vars": [8000.0, 80.0] + [0.0] * 8, "dur": 10.0, "data": [4e-6, 5e-6, 6e-6]},
    ]
    path = tmp_path / "x.awf"
    _write_awf(path, recs, rz=rz)
    out = awfread(str(path))
    assert [r["Var1"] for r in out["recs"]] == [8000.0, 8000.0]
    assert [r["Var2"] for r in out["recs"]] == [90.0, 80.0]
    assert out["recs"][1]["data"].tolist() == pytest.approx([4e-6, 5e-6, 6e-6])
    assert out["recs"][0]["VarName1"] == "Freq"
    assert out["recs"][0]["dur_ms"] == pytest.approx(10.0)


def test_awfread_rejects_truncated_file(tmp_path):
    recs = [{"vars": [8000.0, 90.0] + [0.0] * 8, "dur": 10.0, "data": [1e-6] * 5}]
    path = tmp_path / "x.awf"
    _write_awf(path, recs)
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError, match="not a valid"):
        awfread(str(path))


def test_awfread_rejects_garbage(tmp_path):
    path = tmp_path / "x.awf"
    path.write_bytes(b"not an awf")
    with pytest.raises(ValueError, match="not a valid"):
        awfread(str(path))


def test_upload_awf_gives_abra_dataframe(tmp_path, session_state):
    recs = [
        {"vars": [8000.0, 90.0] + [0.0] * 8, "dur": 10.0, "data": [1e-6] * 244},
        {"vars": [8000.0, 80.0] + [0.0] * 8, "dur": 10.0, "data": [2e-6] * 244},
    ]
    path = tmp_path / "mouse_8kHz.awf"
    _write_awf(path, recs)
    dfs, duration = process_uploaded_files_cached([UploadedFile(path)], "RZ", False, False)
    df = dfs[0]
    assert df.name == "mouse_8kHz.awf"
    assert list(df["Freq(Hz)"]) == [8000.0, 8000.0]
    assert list(df["Level(dB)"]) == [90.0, 80.0]
    assert df.shape[1] == 2 + 244
    assert df["0"].tolist() == pytest.approx([1.0, 2.0])  # volts -> microvolts
    assert duration == pytest.approx(10.0)


def test_upload_awf_attenuation_and_click_modes(tmp_path, session_state):
    recs = [{"vars": [30.0, 0.0] + [0.0] * 8, "dur": 10.0, "data": [1e-6] * 244}]
    path = tmp_path / "click.awf"
    _write_awf(path, recs)
    dfs, _ = process_uploaded_files_cached([UploadedFile(path)], "RZ", True, True)
    assert list(dfs[0]["Freq(Hz)"]) == ["Click"]
    assert list(dfs[0]["PostAtten(dB)"]) == [30.0]

