# Full Native Case39 Feasibility

| item | value |
|---|---|
| `mixed_bank_fit_native.npy` exists | True |
| `mixed_bank_eval_native.npy` exists | True |
| fit native is symlink | False |
| eval native is symlink | False |
| canonical fit resolves to | `/home/pang/projects/DDET-MTD-q1-case39/metric/case14/mixed_bank_fit.npy` |
| canonical eval resolves to | `/home/pang/projects/DDET-MTD-q1-case39/metric/case14/mixed_bank_eval.npy` |
| canonical fit equals case14 sha | True |
| canonical eval equals case14 sha | True |
| existing localretune uses native banks | True |

## Execution Cost

- Full native train/val is already executable: the native fit/eval banks exist and current localretune manifests use them.
- Replacing canonical `metric/case39/mixed_bank_fit.npy` and `mixed_bank_eval.npy` requires changing the canonical links/files to the native fit/eval banks, then rerunning confirm with an explicit STAMP.
- If raw measurement regeneration is required, round1 `case39_reality_check.json` estimated about 4.72 CPU hours for full case39 measurement generation in the audit environment. If using existing native banks, confirm-only reruns should be much cheaper and dominated by scheduler simulation.

## Shortest Safe Route

1. Create a STAMP before touching canonical files.
2. Hash canonical case14 and case39 clean/attack/train/val banks.
3. Point canonical case39 fit/eval to native `metric/case39_localretune/*_native.npy` or generate a new manifest that references them explicitly.
4. Rerun only the existing confirm pipeline with fixed `oracle_protected_ec` and the frozen regime.
5. Run postrun audit and anti-write checks against both current repo `metric/case14` and old repo `metric/case14`.
