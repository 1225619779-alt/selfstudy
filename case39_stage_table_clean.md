| stage | role | slot | recall | unnecessary | cost | served_ratio | stage_meta | label_meta | reference_label | native_case39_stage |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| transfer_frozen_dev | main_result | 1 | 0.0916625 | 13.5 | 0.388106625 | 0.5612625 | None | None | None | native_clean_attack_test_with_frozen_case14_dev |
| transfer_frozen_dev | main_result | 2 | 0.177225 | 17.5 | 0.5574813750000001 | 0.7422124999999999 | None | None | None | native_clean_attack_test_with_frozen_case14_dev |
| source_fixed_replay | mechanism_isolation | 1 | 0.0628875 | 7.875 | 0.280820375 | 0.4529375 | case39_source_fixed_case14winner_native_test | None | None | None |
| source_fixed_replay | mechanism_isolation | 2 | 0.12639999999999998 | 11.5 | 0.457006 | 0.6903375 | case39_source_fixed_case14winner_native_test | None | None | None |
| source_anchor | repair_attempt | 1 | 0.017475 | 0.125 | 0.043281625 | 0.0748875 | case39_source_anchored_localretune | None | None | None |
| source_anchor | repair_attempt | 2 | 0.0542875 | 1.125 | 0.185487375 | 0.3007125 | case39_source_anchored_localretune | None | None | None |
| local_protected | protocol_internal_negative_control | 1 | 0.0155125 | 0 | 0.0370815 | 0.0574 | case39_localretune_protocol_compliant_oracle_protected_ec | None | None | None |
| local_protected | protocol_internal_negative_control | 2 | 0.0156125 | 0 | 0.040386625 | 0.06775 | case39_localretune_protocol_compliant_oracle_protected_ec | None | None | None |
| local_unconstrained | stress_test_out_of_protocol | 1 | 0.0 | 0.0 | 0.0004025 | 0.0055000000000000005 | phase3_oracle_upgrade_confirm | case39_fully_native_localretune | native_clean_attack_test_with_frozen_case14_dev | None |
| local_unconstrained | stress_test_out_of_protocol | 2 | 0.0832625 | 0.25 | 0.18699025000000002 | 0.3188375 | phase3_oracle_upgrade_confirm | case39_fully_native_localretune | native_clean_attack_test_with_frozen_case14_dev | None |
