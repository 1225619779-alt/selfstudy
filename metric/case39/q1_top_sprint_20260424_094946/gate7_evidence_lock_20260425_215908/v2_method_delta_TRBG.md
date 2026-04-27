# V2 Method Delta: TRBG

TRBG-source keeps the source-frozen score weights and regime fixed. It adds a low-dimensional predicted-burden penalty selected on source train/val: `S_guard = S_source - alpha * Bhat`, with locked `alpha=1.0` and `fail_cap_quantile=1.00`. It does not retrain the detector, does not change backend MTD solving, and does not use holdout outcomes for parameter selection.
