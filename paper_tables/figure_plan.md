# Figure and Table Plan

## Main-text figures
1. Figure 1: Online decision pipeline
   - Source: diagram drawn from method description
   - Message: detector -> verification gate -> queue/server -> backend MTD -> metrics

2. Figure 2: Case14 main blind-confirm comparison
   - Source: `case14_confirm_main.csv`
   - Methods: oracle_protected_ec, phase3_proposed, best_threshold, topk_expected_consequence
   - Metrics: recall, unnecessary, cost for slot1 and slot2

3. Figure 3: Case14 ablation / external baseline
   - Sources: `case14_ablation.csv`, `case14_external_main_table_rows_full.csv`
   - Message: fused vs protected; external static and aggressive baselines

4. Figure 4: Case39 five-stage ladder
   - Source: `case39_stage_ladder.csv`
   - Stages: transfer_frozen_dev, source_fixed_replay, source_anchor, local_protected, local_unconstrained
   - Metrics: oracle recall, oracle unnecessary, oracle cost

5. Figure 5: Trustworthiness and runtime
   - Sources: `case14_reliability_summary.csv`, `case39_measure_support.csv`
   - Message: PASS/PASS/PASS + exact-match audit + runtime benchmark

## Main-text tables
1. Table 1: Protocol and metrics dictionary
2. Table 2: Case14 main confirm table
3. Table 3: Case14 ablation / external baseline table
4. Table 4: Case39 stage ladder table

## Supplementary tables
- Significance CI table
- Family breakdown tables
- Import / provenance manifest summary
