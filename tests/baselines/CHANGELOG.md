# Baseline changelog

Every intentional change to expected results is recorded here, newest last.
`pytest --update-baselines` appends a stub entry; fill in the reason.

## 2026-10-26

Reason: initial baselines, recorded from `main` at 9ebe61b (peak finding with
optional pretrained peak-I model, tunable max peak time, background subtraction,
configurable number of peaks). Presets: `default` (UI defaults), `anchor_pk1`
(pretrained peak-I model off), `bk_subtraction` (background subtraction on).
These are the app's current outputs, not expert-verified ground truth.

## 2026-10-26

Reason: added a baseline for the peaks table of `20250616 WT.arf` (default
preset), recorded from the same code as the baselines above.

- `table_peaks_wt_arf`: created
