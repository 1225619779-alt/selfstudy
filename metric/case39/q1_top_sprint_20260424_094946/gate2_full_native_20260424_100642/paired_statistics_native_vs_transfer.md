# Paired Statistics: Native vs Transfer

Deltas are left minus right over the same 8 holdouts. Wins/losses/ties use metric direction.

| comparison | B | metric | mean_delta | CI | sign_p | flip_p | W/L/T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source_frozen_transfer_vs_full_native_safeguarded_retune | 1 | recall | 0.0761 | [0.0559, 0.0970] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_transfer_vs_full_native_safeguarded_retune | 1 | cost | 0.3510 | [0.2899, 0.4171] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_transfer_vs_full_native_safeguarded_retune | 1 | backend_fail | 95.3750 | [82.1250, 111.2500] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_transfer_vs_full_native_unconstrained_retune | 1 | recall | 0.0917 | [0.0713, 0.1124] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_transfer_vs_full_native_unconstrained_retune | 1 | cost | 0.3877 | [0.3267, 0.4546] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_transfer_vs_full_native_unconstrained_retune | 1 | backend_fail | 102.1250 | [89.6250, 116.7500] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_transfer_vs_anchored_retune | 1 | recall | 0.0742 | [0.0556, 0.0930] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_transfer_vs_anchored_retune | 1 | cost | 0.3448 | [0.2874, 0.4080] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_transfer_vs_anchored_retune | 1 | backend_fail | 92.3750 | [79.9969, 107.0000] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_transfer_vs_winner_replay | 1 | recall | 0.0288 | [0.0173, 0.0430] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_transfer_vs_winner_replay | 1 | cost | 0.1073 | [0.0854, 0.1307] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_transfer_vs_winner_replay | 1 | backend_fail | 24.8750 | [17.7500, 33.6250] | 0.0078 | 0.0078 | 0/8/0 |
| phase3_oracle_upgrade_vs_phase3_proposed | 1 | recall | -0.0009 | [-0.0022, 0.0000] | 0.5000 | 0.5000 | 0/2/6 |
| phase3_oracle_upgrade_vs_phase3_proposed | 1 | cost | -0.0046 | [-0.0078, -0.0016] | 0.2188 | 0.0625 | 5/1/2 |
| phase3_oracle_upgrade_vs_phase3_proposed | 1 | backend_fail | -0.8750 | [-2.0000, 0.0000] | 0.3750 | 0.2500 | 4/1/3 |
| phase3_oracle_upgrade_vs_topk_expected_consequence | 1 | recall | -0.0723 | [-0.0872, -0.0584] | 0.0078 | 0.0078 | 0/8/0 |
| phase3_oracle_upgrade_vs_topk_expected_consequence | 1 | cost | -0.3524 | [-0.4308, -0.2925] | 0.0078 | 0.0078 | 8/0/0 |
| phase3_oracle_upgrade_vs_topk_expected_consequence | 1 | backend_fail | -94.1250 | [-110.2500, -81.0000] | 0.0078 | 0.0078 | 8/0/0 |
| best_full_native_operating_point_vs_source_frozen_transfer | 1 | recall | -0.0194 | [-0.0340, -0.0052] | 0.4531 | 0.0625 | 2/5/1 |
| best_full_native_operating_point_vs_source_frozen_transfer | 1 | cost | -0.0353 | [-0.0743, 0.0019] | 0.7266 | 0.1328 | 5/3/0 |
| best_full_native_operating_point_vs_source_frozen_transfer | 1 | backend_fail | -8.0000 | [-13.0000, -2.2500] | 0.0703 | 0.0469 | 7/1/0 |
| source_frozen_transfer_vs_full_native_safeguarded_retune | 2 | recall | 0.1616 | [0.1364, 0.1820] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_transfer_vs_full_native_safeguarded_retune | 2 | cost | 0.5171 | [0.4164, 0.6172] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_transfer_vs_full_native_safeguarded_retune | 2 | backend_fail | 121.6250 | [104.1219, 141.8750] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_transfer_vs_full_native_unconstrained_retune | 2 | recall | 0.0939 | [0.0663, 0.1209] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_transfer_vs_full_native_unconstrained_retune | 2 | cost | 0.3705 | [0.2839, 0.4529] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_transfer_vs_full_native_unconstrained_retune | 2 | backend_fail | 83.8750 | [69.0000, 102.2500] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_transfer_vs_anchored_retune | 2 | recall | 0.1229 | [0.0958, 0.1463] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_transfer_vs_anchored_retune | 2 | cost | 0.3720 | [0.2858, 0.4526] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_transfer_vs_anchored_retune | 2 | backend_fail | 82.0000 | [65.6250, 101.7500] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_transfer_vs_winner_replay | 2 | recall | 0.0508 | [0.0343, 0.0656] | 0.0078 | 0.0078 | 8/0/0 |
| source_frozen_transfer_vs_winner_replay | 2 | cost | 0.1005 | [0.0661, 0.1302] | 0.0078 | 0.0078 | 0/8/0 |
| source_frozen_transfer_vs_winner_replay | 2 | backend_fail | 12.2500 | [6.6250, 18.6250] | 0.0078 | 0.0078 | 0/8/0 |
| phase3_oracle_upgrade_vs_phase3_proposed | 2 | recall | 0.0824 | [0.0629, 0.1045] | 0.0078 | 0.0078 | 8/0/0 |
| phase3_oracle_upgrade_vs_phase3_proposed | 2 | cost | 0.1766 | [0.1319, 0.2183] | 0.0078 | 0.0078 | 0/8/0 |
| phase3_oracle_upgrade_vs_phase3_proposed | 2 | backend_fail | 42.3750 | [31.6219, 54.2500] | 0.0078 | 0.0078 | 0/8/0 |
| phase3_oracle_upgrade_vs_topk_expected_consequence | 2 | recall | -0.0489 | [-0.0623, -0.0352] | 0.0078 | 0.0078 | 0/8/0 |
| phase3_oracle_upgrade_vs_topk_expected_consequence | 2 | cost | -0.3090 | [-0.3664, -0.2512] | 0.0078 | 0.0078 | 8/0/0 |
| phase3_oracle_upgrade_vs_topk_expected_consequence | 2 | backend_fail | -77.7500 | [-93.2500, -64.7500] | 0.0078 | 0.0078 | 8/0/0 |
| best_full_native_operating_point_vs_source_frozen_transfer | 2 | recall | -0.0451 | [-0.0624, -0.0286] | 0.0078 | 0.0078 | 0/8/0 |
| best_full_native_operating_point_vs_source_frozen_transfer | 2 | cost | -0.0615 | [-0.0927, -0.0243] | 0.0703 | 0.0312 | 7/1/0 |
| best_full_native_operating_point_vs_source_frozen_transfer | 2 | backend_fail | -6.1250 | [-10.7500, -1.2500] | 0.4531 | 0.0625 | 5/2/1 |
