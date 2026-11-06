"""Analysis settings included in table exports."""

import csv
import io
import subprocess
from pathlib import Path


def code_version():
    """Identify the checkout, including changes after a tag or uncommitted edits."""
    root = Path(__file__).resolve().parent.parent
    if not (root / ".git").exists():
        return "Unavailable (not a Git checkout)"
    try:
        return subprocess.run(
            ["git", "-C", str(root), "describe", "--tags", "--always", "--long", "--dirty"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Unavailable ({type(exc).__name__})"


def analysis_metadata(settings, *, peaks):
    metadata = {
        "ABRA code version": code_version(),
        "Input time scale (ms)": settings.get("time_scale", 10.0),
        "Input units": settings.get("units", "Microvolts"),
        "Y multiplier": settings.get("multiply_y_factor", 1.0),
        "Input levels are attenuation": settings.get("atten", False),
    }
    if peaks:
        pretrained = settings.get("pretrained_pk1", True)
        background = settings.get("pk_bk_subtraction", False)
        metadata.update({
            "Peak I method": "Pretrained model" if pretrained else "Search around time",
            "Peak I search time (ms)": "" if pretrained else settings.get("pk1_anchor", 1.0),
            "Number of peaks": settings.get("num_peaks", 5),
            "Peak cutoff time (ms)": settings.get("max_pk_time", 8.0),
            "Background subtraction": background,
            "Background window (ms)": settings.get("bk_window", 1.0) if background else "",
            "Peaks below threshold": settings.get("peaks_below_thresh", False),
            "All peaks in table": settings.get("all_peaks", False),
        })
    return metadata


def add_analysis_metadata(table, settings, *, peaks, manual_edits=None):
    table.attrs["analysis_metadata"] = analysis_metadata(settings, peaks=peaks)
    if manual_edits is not None:
        table.attrs["manual_edits"] = manual_edits
    return table


def table_with_settings_csv(table, metadata, manual_edits=None):
    """Create a CSV with a settings section followed by the normal data table."""
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(metadata.items())
    writer.writerow([])
    export_table = table.copy()
    if manual_edits:
        for column, values in manual_edits.items():
            export_table[column] = [values[index] for index in table.index]
    export_table.to_csv(output, index=False, lineterminator="\n")
    return output.getvalue().encode("utf-8-sig")
