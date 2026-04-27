# Gate9 Fresh Pareto Frontier

| method | B | recall-cost efficient | recall-backend efficient | dominated_by |
| --- | ---: | --- | --- | --- |
| TRBG-source | 1 | True | True |  |
| TRBG-source | 2 | False | False | TRFG-native-fail |
| TRFG-native-fail | 1 | False | False | TRBG-source;TRFG-source |
| TRFG-native-fail | 2 | True | True |  |
| TRFG-source | 1 | True | True |  |
| TRFG-source | 2 | False | False | TRFG-native-fail |
| native_safeguarded_retune | 1 | True | True |  |
| native_safeguarded_retune | 2 | True | True |  |
| native_unconstrained_retune | 1 | True | True |  |
| native_unconstrained_retune | 2 | True | True |  |
| source_frozen_transfer | 1 | False | False | TRFG-source |
| source_frozen_transfer | 2 | False | False | TRFG-native-fail;TRFG-source |
| topk_expected_consequence | 1 | False | False | TRBG-source;TRFG-native-fail;TRFG-source;winner_replay |
| topk_expected_consequence | 2 | False | False | TRBG-source;TRFG-native-fail;TRFG-source;source_frozen_transfer;winner_replay |
| winner_replay | 1 | True | True |  |
| winner_replay | 2 | True | True |  |
