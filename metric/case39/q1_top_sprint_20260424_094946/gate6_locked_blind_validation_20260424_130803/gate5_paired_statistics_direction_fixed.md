# Gate 5 Paired Statistics Direction Fixed

Mean deltas and bootstrap CIs are preserved from Gate 5; only W/L/T direction is corrected for lower-is-better metrics.

| B | comparison | metric | direction | old | fixed | flag |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | TRBG-source vs source_frozen_transfer | backend_fail | lower | 2/6/0 | 6/2/0 | True |
| 1 | TRBG-source vs source_frozen_transfer | cost | lower | 3/5/0 | 5/3/0 | True |
| 1 | TRBG-source vs source_frozen_transfer | unnecessary | lower | 0/7/1 | 7/0/1 | True |
| 1 | TRBG-native-burden vs source_frozen_transfer | backend_fail | lower | 3/5/0 | 5/3/0 | True |
| 1 | TRBG-native-burden vs source_frozen_transfer | cost | lower | 5/3/0 | 3/5/0 | True |
| 1 | TRBG-native-burden vs source_frozen_transfer | unnecessary | lower | 0/7/1 | 7/0/1 | True |
| 1 | TRBG-native-burden vs source_frozen_transfer | recover_fail | lower | 4/1/3 | 1/4/3 | True |
| 1 | TRBG-source vs topk_expected_consequence | backend_fail | lower | 3/5/0 | 5/3/0 | True |
| 1 | TRBG-source vs topk_expected_consequence | cost | lower | 2/6/0 | 6/2/0 | True |
| 1 | TRBG-source vs topk_expected_consequence | unnecessary | lower | 1/7/0 | 7/1/0 | True |
| 1 | TRBG-source vs topk_expected_consequence | recover_fail | lower | 4/2/2 | 2/4/2 | True |
| 1 | TRBG-native-burden vs topk_expected_consequence | backend_fail | lower | 6/2/0 | 2/6/0 | True |
| 1 | TRBG-native-burden vs topk_expected_consequence | unnecessary | lower | 1/7/0 | 7/1/0 | True |
| 1 | TRBG-native-burden vs topk_expected_consequence | recover_fail | lower | 4/1/3 | 1/4/3 | True |
| 1 | TRBG-source vs winner_replay | backend_fail | lower | 7/0/1 | 0/7/1 | True |
| 1 | TRBG-source vs winner_replay | cost | lower | 6/2/0 | 2/6/0 | True |
| 1 | TRBG-source vs winner_replay | unnecessary | lower | 3/4/1 | 4/3/1 | True |
| 1 | TRBG-source vs winner_replay | recover_fail | lower | 5/2/1 | 2/5/1 | True |
| 1 | TRBG-native-burden vs winner_replay | backend_fail | lower | 8/0/0 | 0/8/0 | True |
| 1 | TRBG-native-burden vs winner_replay | cost | lower | 7/1/0 | 1/7/0 | True |
| 1 | TRBG-native-burden vs winner_replay | unnecessary | lower | 4/3/1 | 3/4/1 | True |
| 1 | TRBG-native-burden vs winner_replay | recover_fail | lower | 6/0/2 | 0/6/2 | True |
| 2 | TRBG-source vs source_frozen_transfer | backend_fail | lower | 2/6/0 | 6/2/0 | True |
| 2 | TRBG-source vs source_frozen_transfer | cost | lower | 2/6/0 | 6/2/0 | True |
| 2 | TRBG-source vs source_frozen_transfer | unnecessary | lower | 0/7/1 | 7/0/1 | True |
| 2 | TRBG-source vs source_frozen_transfer | recover_fail | lower | 2/4/2 | 4/2/2 | True |
| 2 | TRBG-native-burden vs source_frozen_transfer | backend_fail | lower | 1/7/0 | 7/1/0 | True |
| 2 | TRBG-native-burden vs source_frozen_transfer | cost | lower | 2/6/0 | 6/2/0 | True |
| 2 | TRBG-native-burden vs source_frozen_transfer | unnecessary | lower | 0/7/1 | 7/0/1 | True |
| 2 | TRBG-native-burden vs source_frozen_transfer | recover_fail | lower | 2/3/3 | 3/2/3 | True |
| 2 | TRBG-source vs topk_expected_consequence | backend_fail | lower | 5/3/0 | 3/5/0 | True |
| 2 | TRBG-source vs topk_expected_consequence | cost | lower | 5/3/0 | 3/5/0 | True |
| 2 | TRBG-source vs topk_expected_consequence | unnecessary | lower | 0/7/1 | 7/0/1 | True |
| 2 | TRBG-source vs topk_expected_consequence | recover_fail | lower | 3/4/1 | 4/3/1 | True |
| 2 | TRBG-native-burden vs topk_expected_consequence | unnecessary | lower | 0/7/1 | 7/0/1 | True |
| 2 | TRBG-source vs winner_replay | backend_fail | lower | 5/3/0 | 3/5/0 | True |
| 2 | TRBG-source vs winner_replay | cost | lower | 6/2/0 | 2/6/0 | True |
| 2 | TRBG-source vs winner_replay | unnecessary | lower | 3/4/1 | 4/3/1 | True |
| 2 | TRBG-source vs winner_replay | recover_fail | lower | 4/1/3 | 1/4/3 | True |
| 2 | TRBG-native-burden vs winner_replay | backend_fail | lower | 6/2/0 | 2/6/0 | True |
| 2 | TRBG-native-burden vs winner_replay | cost | lower | 7/1/0 | 1/7/0 | True |
| 2 | TRBG-native-burden vs winner_replay | unnecessary | lower | 2/5/1 | 5/2/1 | True |
| 2 | TRBG-native-burden vs winner_replay | recover_fail | lower | 5/1/2 | 1/5/2 | True |

Rows with possible original direction mismatch: `43`.
