# Gate9 Case39 Original 8-Holdout Secondary Pareto

Secondary diagnostic only; not used for method replacement.

| method | B | recall_cost_efficient | recall_backend_efficient | dominated_by |
| --- | --- | --- | --- | --- |
| TRBG-source | 1 | True | False | TRFG-native-fail |
| TRBG-source | 2 | False | False | TRFG-native-fail |
| TRFG-native-fail | 1 | True | True |  |
| TRFG-native-fail | 2 | True | True |  |
| TRFG-source | 1 | False | False | TRFG-native-fail |
| TRFG-source | 2 | True | True |  |
| native_safeguarded_retune | 1 | True | True |  |
| native_safeguarded_retune | 2 | True | True |  |
| native_unconstrained_retune | 1 | True | True |  |
| native_unconstrained_retune | 2 | True | True |  |
| source_frozen_transfer | 1 | False | False | TRBG-source;TRFG-native-fail;TRFG-source |
| source_frozen_transfer | 2 | False | False | TRFG-source |
| topk_expected_consequence | 1 | False | False | TRBG-source;TRFG-native-fail |
| topk_expected_consequence | 2 | False | False | TRBG-source;TRFG-native-fail;TRFG-source |
| winner_replay | 1 | True | True |  |
| winner_replay | 2 | True | False | TRFG-native-fail |
