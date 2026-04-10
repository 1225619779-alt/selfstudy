# Claim recommendation

Primary result: use **native_clean_attack_test_with_frozen_case14_dev** as the main paper result.

Interpretation: native case39 local selection drifts toward over-conservative policies, while source-frozen transfer acts like regularization and yields the most useful risk–cost trade-off.

Recommended stage roles:
- Main result: transfer_frozen_dev
- Mechanism/isolation: source_fixed_replay
- Repair attempt: source_anchor
- Protocol-compliant negative control: local_protected
- Protocol-violating stress test: local_unconstrained

Key oracle recalls (slot1 / slot2):
- transfer_frozen_dev: None / None
- source_fixed_replay: 0.0628875 / 0.12639999999999998
- source_anchor: 0.017475 / 0.0542875
- local_protected: 0.0155125 / 0.0156125
- local_unconstrained: None / None

Next best step is not another heavy run. Focus on paper packaging, significance presentation, and artifact cleanup.
