# Proxy Robustness Summary

`success_burden_proxy` uses alpha `0.5000` derived from native train/val medians; no test holdout was used to tune alpha.

For `burden_proxy`, rank score is inverted burden, so higher is lower served burden rather than higher consequence recall.

| proxy | B | source_score | source_vs_topk | source_vs_winner | source_vs_safeguarded | rho_vs_product | source_cost | source_backend |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| product_proxy | 1 | 0.0917 | 0.0194 | 0.0288 | 0.0761 | 1.0000 | 0.3881 | 102.7500 |
| product_proxy | 2 | 0.1772 | 0.0451 | 0.0508 | 0.1616 | 1.0000 | 0.5575 | 130.1250 |
| additive_proxy | 1 | 0.0872 | 0.0187 | 0.0259 | 0.0729 | 1.0000 | 0.3881 | 102.7500 |
| additive_proxy | 2 | 0.1643 | 0.0396 | 0.0444 | 0.1501 | 1.0000 | 0.5575 | 130.1250 |
| backend_success_proxy | 1 | 0.2998 | 0.0746 | 0.0983 | 0.2495 | 1.0000 | 0.3881 | 102.7500 |
| backend_success_proxy | 2 | 0.5600 | 0.1489 | 0.1620 | 0.5098 | 1.0000 | 0.5575 | 130.1250 |
| recovery_aware_proxy | 1 | 0.0956 | 0.0201 | 0.0300 | 0.0793 | 1.0000 | 0.3881 | 102.7500 |
| recovery_aware_proxy | 2 | 0.1850 | 0.0471 | 0.0531 | 0.1686 | 1.0000 | 0.5575 | 130.1250 |
| burden_proxy | 1 | -0.7285 | -0.0721 | -0.1771 | -0.6659 | -1.0000 | 0.3881 | 102.7500 |
| burden_proxy | 2 | -0.9739 | -0.0751 | -0.1186 | -0.9019 | -1.0000 | 0.5575 | 130.1250 |
| success_burden_proxy | 1 | -7.6374 | -0.5211 | -1.8505 | -7.1877 | -1.0000 | 0.3881 | 102.7500 |
| success_burden_proxy | 2 | -9.8270 | -0.5810 | -1.0879 | -9.3014 | -0.9277 | 0.5575 | 130.1250 |

## Stability

- Source-frozen vs topk is stable on high-is-better consequence proxies excluding pure burden: `False`.
- Source-frozen vs winner_replay is stable on high-is-better consequence proxies excluding pure burden: `False`.
- Native safeguarded collapse persists across product/additive/backend-success/recovery-aware proxies: `True`.
- Pure burden scoring penalizes source-frozen because it is a high-service/high-backend-fail operating point.
