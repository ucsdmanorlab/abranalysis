"""Golden-baseline helpers.

A baseline is a JSON file in tests/baselines/ holding results that the app
produced on the demo data. Tests compare fresh results to it (with
tolerances). When an algorithm is changed on purpose, run

    pytest --update-baselines

to rewrite the files; a summary of what moved is printed and a stub entry is
appended to tests/baselines/CHANGELOG.md for the author to complete.
"""
import datetime
import json
import math
from pathlib import Path

import numpy as np

BASELINE_DIR = Path(__file__).parent / "baselines"


def to_jsonable(obj):
    """Convert numpy / pandas scalars and containers to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return None if math.isnan(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def diff_values(expected, actual, path="", rtol=1e-4, atol=1e-6):
    """Return a list of human-readable differences between two JSON-like trees."""
    diffs = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            sub = f"{path}/{key}"
            if key not in actual:
                diffs.append(f"{sub}: missing from new results (was {expected[key]!r})")
            elif key not in expected:
                diffs.append(f"{sub}: new entry {actual[key]!r}")
            else:
                diffs.extend(diff_values(expected[key], actual[key], sub, rtol, atol))
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: length {len(expected)} -> {len(actual)} ({expected!r} -> {actual!r})")
        else:
            for i, (e, a) in enumerate(zip(expected, actual)):
                diffs.extend(diff_values(e, a, f"{path}[{i}]", rtol, atol))
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)) \
            and not isinstance(expected, bool) and not isinstance(actual, bool):
        if not math.isclose(expected, actual, rel_tol=rtol, abs_tol=atol):
            diffs.append(f"{path}: {expected!r} -> {actual!r}")
    elif expected != actual:
        diffs.append(f"{path}: {expected!r} -> {actual!r}")
    return diffs


def check_baseline(name, results, update, rtol=1e-4, atol=1e-6, report=None):
    """Compare ``results`` to baselines/<name>.json, or rewrite it when ``update``."""
    path = BASELINE_DIR / f"{name}.json"
    actual = to_jsonable(results)

    if update:
        old = json.loads(path.read_text()) if path.exists() else None
        diffs = [] if old is None else diff_values(old, actual, rtol=rtol, atol=atol)
        BASELINE_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
        if report is not None:
            report.append((name, "created" if old is None else "updated", diffs))
        return

    if not path.exists():
        raise AssertionError(
            f"No baseline '{name}'. If this is a new test, create it with: pytest --update-baselines"
        )
    expected = json.loads(path.read_text())
    diffs = diff_values(expected, actual, rtol=rtol, atol=atol)
    if diffs:
        shown = "\n  ".join(diffs[:40])
        more = f"\n  ... and {len(diffs) - 40} more" if len(diffs) > 40 else ""
        raise AssertionError(
            f"Results no longer match baseline '{name}' ({len(diffs)} differences):\n  {shown}{more}\n"
            "If you changed an algorithm on purpose, run `pytest --update-baselines`, "
            "review the diff, and explain it in the PR."
        )


def write_changelog(report):
    """Append a stub entry describing which baselines moved."""
    changed = [(n, k, d) for n, k, d in report if k == "created" or d]
    if not changed:
        return
    lines = [f"\n## {datetime.date.today().isoformat()}", "", "Reason: _TODO - explain why these values changed_", ""]
    for name, kind, diffs in changed:
        lines.append(f"- `{name}`: {kind}" + (f", {len(diffs)} values changed" if diffs else ""))
    changelog = BASELINE_DIR / "CHANGELOG.md"
    if not changelog.exists():
        changelog.write_text("# Baseline changelog\n\nEvery intentional change to expected results is recorded here.\n")
    with changelog.open("a") as fh:
        fh.write("\n".join(lines) + "\n")
