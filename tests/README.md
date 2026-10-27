# Testing ABRA

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest                      # everything (~1-2 min)
pytest -m "not slow"        # quick checks only
pytest tests/test_baselines.py   # just the golden baselines
```

## What the tests protect

| File | Purpose |
|---|---|
| `test_imports.py` | Every module imports. |
| `test_file_parsing.py` | `.arf`/`.csv`/`.tsv`/`.asc` parsing and the upload pipeline. |
| `test_peak_and_threshold_contracts.py` | **Tier A.** Properties that must hold for *any* correct algorithm (e.g. peaks are ordered, thresholds fall on measured levels). Do not change these when you improve the algorithm. |
| `test_baselines.py` | **Tier B.** Exact thresholds and peaks the app currently produces for the demo files, under several settings presets. |
| `test_session_logic.py`, `test_plotting.py`, `test_app_e2e.py` | Tables, manual edits, figures, app startup, and settings-change handling. |

## Two kinds of numeric tests

**Most changes (new file types, UI, refactors) must not change any threshold or peak.**
If a baseline test fails, you changed behaviour. Fix your code.

**Some changes intentionally improve the algorithm.** Then the baselines should change:

1. Run `pytest --update-baselines`. This rewrites `tests/baselines/*.json`.
2. Review the diff (`git diff tests/baselines`). Every changed number should be one you expected.
3. Fill in the new entry that was appended to `tests/baselines/CHANGELOG.md`: why the values changed.
4. Have someone who knows ABR (not just the person who wrote the code) approve the baseline diff in the PR.

Baselines are the app's current output, **not** proof of correctness. A baseline can be wrong.

## Known-wrong behaviour

If you find a bug, write a test for the correct behaviour and mark it
`@pytest.mark.xfail(strict=True, reason="...")`. It documents the bug without failing CI,
and fails loudly (strict) once someone fixes it so the marker gets removed.

## Adding tests

- Use the shared fixtures in `tests/conftest.py`: `session_state` / `apply_settings` (fake `st.session_state`), `load_demo` (runs the real upload pipeline on files in `demo_data`), `make_abr` (synthetic waveforms).
- Prefer small deterministic synthetic waveforms for contracts; use demo files for baselines.
- Tests run from the repo root (conftest handles this).

## Cross-platform note

Baselines were generated on macOS. Indices and thresholds are compared exactly, floats with `rtol=1e-4`.
If CI on Linux disagrees with a local result, investigate before regenerating.
