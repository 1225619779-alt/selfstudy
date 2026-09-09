# Fixed-record scheduling reproduction package

This package reproduces the five-method historical scheduling comparison and a
bounded post-hoc parameter diagnostic. It runs without the original repository,
private paths, GPU, model checkpoint, or grid simulator. Inputs are included.

## Run

Download `Supplementary_Replay_Package_R1.zip` from this directory and extract it
into a new folder. Run the commands below inside the extracted folder. The
individually browsable Python files are byte-identical mirrors for source review;
the ZIP supplies the input banks and all support files as well.

Use Python 3.11 or newer with NumPy and SciPy. Python 3.12 was exercised locally;
other Python versions were not independently tested. The verified package
versions are recorded in `requirements.txt` and `run_001/RUN_CONTEXT.json`.
Install them in your own environment, not in an active research environment.

```text
python -m unittest test_contracts -v
python reproduce.py --out my_new_run
python analyze.py --run my_new_run --out my_new_analysis
```

Use new output names: existing directories are never replaced. A normal full run
contains 80 baseline traces and 144 one-at-a-time sensitivity traces. The saved
`run_001` contains saved execution summaries, not the larger per-trace NPZ outputs;
a fresh `reproduce.py` invocation writes those outputs to the chosen new folder.
To inspect the supplied summaries without running any queue or MILP, use:

```text
python analyze.py --run run_001 --out saved_summary_check
```

## Scope and lineage

- `data/BANK.npz`: all 8 x 540 endpoint records and 1,903 candidate occurrences,
  including saved forecasts and evaluator-only outcomes. The field
  `prediction_origin` records reconstruction from frozen calibration inputs.
- `data/HISTORICAL_REFERENCE.npz`: saved P02/P05C start times, terminal states and
  metrics, used for comparison after execution, never by the policy.
- `data/CONFIG.json`: all S0 configurations, common gate and exact input hashes.
- `data/frozen_binned.npz`, `frozen_scores.py`, `fusion_copy.py`: seven frozen
  scalar calibration models and lookup/fusion arithmetic. Cross-platform fusion
  arithmetic has tiny floating-point differences; field-by-field differences are
  recorded in `data/FROZEN_LOOKUP_REPRODUCTION.json`. Scheduling reproductions consume
  the original saved forecasts unchanged, not these recomputed values. No
  detector training, calibration fitting, recovery or physical call is needed.
- `policies_p02_copy.py`: unchanged historical engine for S0 and FIFO variants.
- `p05b_scheduler_core.py`: unchanged ATC and current-queue planning algorithm.
- `planning_replay.py`: original pure `decide`, `check` and `run` functions copied
  verbatim; the old task-specific command-line entry is excluded.
- `reference_audit.py`: P02's independent state/action audit, with fixed score
  coefficients and eligibility constant expressed through the configuration
  solely to check the declared parameter perturbations. No policy changes.
- `reproduce.py`: portable entry, positive-list policy interface and regression
  comparison; `analyze.py`: saved-output calculations only.

The engine and the evaluator retain actual outcomes to release services and score
results. Admission/ranking receive only arrived-job forecasts and active count;
the planning DTO excludes actual duration and future completion time. Any historic
actual-cost-budget admission option is disabled and rejected by this entry.

NPZ uses numeric arrays plus explicit JSON metadata; `allow_pickle=False` is used.
The safe serializer records array dtype, shape and SHA-256. No downloaded pickle,
Notebook or upstream simulation program is executed.

## What this does and does not reproduce

It reproduces the complete **five-method fixed-record comparison** (S0, value-gated
FIFO Q25, all-accept FIFO, ATC-start, rolling planner), not the entire physical
FDIA/MTD pipeline, detector training, every historical P02 menu, or P04 offline
optimization. Existing P02 menu metrics are included only for retrospective cap
accounting. Those scopes must not be conflated in a data-availability statement.

All eight records are already exposed and source-overlapping. These files cannot
be used as a newly independent holdout. Same source index does not imply identical
scenario assignment. Sensitivity is local +/-20%, one parameter at a time, and
post-hoc: there is no new selected controller and no inferential p-value. Realized
resource subsets are outcome-conditioned descriptive subsets, not a newly matched
randomized comparison. Retain all rows, including unfavorable and unchanged ones.

Planner wall-clock time is environment-dependent; a time-limited solve on a slower
machine may fall back as specified. The included reference makes such differences
visible instead of quietly treating changed decisions as an exact reproduction.

## Public release and integrity

This author-authorized public release adds executable scheduling code and saved
candidate inputs to the earlier `kbs-v8-data` result tables. Use the manuscript's
commit-pinned citation to identify the version; a moving branch is not a version.
Check the outer archive against the directory's `SHA256SUMS.txt`, and the extracted
contents against `PACKAGE_SHA256SUMS.txt`. Do not describe this as a complete
simulator, independently acquired test set, or detector-training release.

The accompanying analysis scope preserves the original pre-analysis declaration.
Its historical local paths document provenance; no executable depends on them.
The frozen inputs, numerical outputs and scheduling algorithms are unchanged from
the tested release candidate. Only release instructions and integrity manifests
were added or clarified. No additional software or data license is asserted by
this release; public availability should not be confused with a new license grant.
