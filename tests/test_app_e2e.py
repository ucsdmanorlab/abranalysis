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
