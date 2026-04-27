# Red Flag Register

| red flag | required handling |
| --- | --- |
| canonical case39 fit/eval still resolve to case14 | Do not rely on canonical case39 train/val paths until release cleanup. |
| severity truth target remains ang_no * ang_str proxy | Use proxy-consequence-guided wording, not physical-consequence-aware. |
| recover_fail is not in primary scoring chain | Recovery-aware language is limited to post-hoc robustness diagnostics. |
| Gate6 is not fresh physical-solver validation | Describe as recombined stress replication only. |
| Gate5 W/L/T had lower-is-better direction issue | Gate6 repaired direction; conclusion unchanged. |
| full-solver fresh validation runtime is very high | 8-bank fresh validation is infeasible without checkpoint/resume or reduced protocol. |
| current submitted TPWRS version cannot be replaced mid-review | Use this as rebuttal/resubmission evidence, not a silent submission mutation. |
