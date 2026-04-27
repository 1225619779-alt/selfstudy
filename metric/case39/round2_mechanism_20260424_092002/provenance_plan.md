# Anti-write / Provenance Plan

## Why Current Anti-write Evidence Is Insufficient

- Round1 found no usable current-run STAMP under `/tmp/case39_*`.
- Without a STAMP created before rerun, file mtime checks cannot prove that case14 assets were not modified during the case39 run.
- Existing hash manifests are useful, but they do not replace a before/after write barrier for a new rerun.

## Next Rerun STAMP Protocol

```bash
STAMP=/tmp/case39_round3_$(date +%Y%m%d_%H%M%S).stamp
touch "$STAMP"
# run the selected case39 pipeline
find metric/case14 -type f -newer "$STAMP" -print > anti_write_q1_case14.txt
find /home/pang/projects/DDET-MTD/metric/case14 -type f -newer "$STAMP" -print > anti_write_oldrepo_case14.txt
```

## Files To Hash Before And After

- `metric/case14/metric_clean_alarm_scores_full.npy`
- `metric/case14/metric_attack_alarm_scores_400.npy`
- `metric/case14/mixed_bank_fit.npy`
- `metric/case14/mixed_bank_eval.npy`
- `metric/case39/metric_clean_alarm_scores_full.npy`
- `metric/case39/metric_attack_alarm_scores_400.npy`
- `metric/case39/mixed_bank_fit.npy`
- `metric/case39/mixed_bank_eval.npy`
- `metric/case39_localretune/mixed_bank_fit_native.npy`
- `metric/case39_localretune/mixed_bank_eval_native.npy`
- all manifest JSON files used by confirm

## Proof Standard

- `anti_write_q1_case14.txt` and `anti_write_oldrepo_case14.txt` must be empty.
- Before/after SHA256 for case14 files must match.
- Case39 output manifests must reference case39/native paths for the intended run.
- The final audit bundle must include STAMP path, pre-hash JSON, post-hash JSON, anti-write txt files, and an outputs tree.
