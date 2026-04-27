# Paired Statistics

Deltas are left minus right. `left_wins` follows metric direction: higher is better for recall/served_ratio; lower is better for unnecessary/cost/delay/backend_fail.

| comparison | slot | metric | mean delta | 95% CI | sign p | sign-flip p | W/L/T |
|---|---:|---|---:|---|---:|---:|---:|
| source_frozen_vs_winner_replay | 1 | recall | 0.0288 | [0.0175, 0.0431] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_vs_winner_replay | 1 | cost | 0.1073 | [0.0855, 0.1311] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_vs_winner_replay | 1 | backend_fail | 24.8750 | [18.0000, 33.6250] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_vs_anchored_retune | 1 | recall | 0.0742 | [0.0558, 0.0932] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_vs_anchored_retune | 1 | cost | 0.3448 | [0.2869, 0.4077] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_vs_anchored_retune | 1 | backend_fail | 92.3750 | [79.5000, 106.6250] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_vs_native_safeguarded_retune | 1 | recall | 0.0761 | [0.0561, 0.0971] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_vs_native_safeguarded_retune | 1 | cost | 0.3510 | [0.2886, 0.4164] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_vs_native_safeguarded_retune | 1 | backend_fail | 95.3750 | [81.8750, 111.2500] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_vs_native_unconstrained_retune | 1 | recall | 0.0917 | [0.0718, 0.1126] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_vs_native_unconstrained_retune | 1 | cost | 0.3877 | [0.3267, 0.4531] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_vs_native_unconstrained_retune | 1 | backend_fail | 102.1250 | [89.2500, 116.6250] | 0.0078 | 0.0078 | 0/8/0 |
| oracle_vs_phase3_proposed | 1 | recall | -0.0011 | [-0.0139, 0.0087] | 0.4531 | 0.9219 | 5/2/1 |
| oracle_vs_phase3_proposed | 1 | cost | -0.0380 | [-0.0693, -0.0119] | 0.0703 | 0.0312 | 7/1/0 |
| oracle_vs_phase3_proposed | 1 | backend_fail | -6.8750 | [-9.1250, -4.7500] | 0.0078 | 0.0078 | 8/0/0 |
| oracle_vs_topk_expected_consequence | 1 | recall | -0.0025 | [-0.0179, 0.0101] | 1.0000 | 0.7969 | 4/3/1 |
| oracle_vs_topk_expected_consequence | 1 | cost | -0.0384 | [-0.0741, -0.0059] | 0.2891 | 0.0781 | 6/2/0 |
| oracle_vs_topk_expected_consequence | 1 | backend_fail | -6.7500 | [-10.6250, -3.3750] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_vs_winner_replay | 2 | recall | 0.0508 | [0.0342, 0.0659] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_vs_winner_replay | 2 | cost | 0.1005 | [0.0667, 0.1308] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_vs_winner_replay | 2 | backend_fail | 12.2500 | [6.6250, 18.7531] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_vs_anchored_retune | 2 | recall | 0.1229 | [0.0944, 0.1462] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_vs_anchored_retune | 2 | cost | 0.3720 | [0.2848, 0.4529] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_vs_anchored_retune | 2 | backend_fail | 82.0000 | [65.8750, 101.8750] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_vs_native_safeguarded_retune | 2 | recall | 0.1616 | [0.1364, 0.1820] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_vs_native_safeguarded_retune | 2 | cost | 0.5171 | [0.4181, 0.6180] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_vs_native_safeguarded_retune | 2 | backend_fail | 121.6250 | [104.1250, 141.5000] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_vs_native_unconstrained_retune | 2 | recall | 0.0939 | [0.0658, 0.1210] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_vs_native_unconstrained_retune | 2 | cost | 0.3705 | [0.2839, 0.4524] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_vs_native_unconstrained_retune | 2 | backend_fail | 83.8750 | [69.1250, 102.6250] | 0.0078 | 0.0078 | 0/8/0 |
| oracle_vs_phase3_proposed | 2 | recall | 0.0016 | [-0.0130, 0.0141] | 0.7266 | 0.8203 | 5/3/0 |
| oracle_vs_phase3_proposed | 2 | cost | -0.0209 | [-0.0455, 0.0004] | 0.2891 | 0.1641 | 6/2/0 |
| oracle_vs_phase3_proposed | 2 | backend_fail | -4.8750 | [-8.3750, -1.8750] | 0.0703 | 0.0312 | 7/1/0 |
| oracle_vs_topk_expected_consequence | 2 | recall | 0.0096 | [-0.0006, 0.0183] | 0.7266 | 0.1094 | 5/3/0 |
| oracle_vs_topk_expected_consequence | 2 | cost | -0.0095 | [-0.0258, 0.0074] | 0.7266 | 0.3203 | 5/3/0 |
| oracle_vs_topk_expected_consequence | 2 | backend_fail | -3.8750 | [-6.2500, -1.0000] | 0.2891 | 0.0547 | 6/2/0 |

## Interpretation

- Source-frozen is consistently higher-recall than anchored and safeguarded native retunes, but it pays higher intervention/cost burden.
- Source-frozen versus winner replay separates two effects: source-frozen uses case14-backed train/val, while winner replay uses native train/val with the source winner. A positive recall delta here indicates calibration-bank shift is a real mechanism.
- Oracle versus phase3/topk at source-frozen should be treated as an operating-point comparison, not proof of a new native case39 family.
