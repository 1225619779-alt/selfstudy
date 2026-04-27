# Gate6 Statistics Direction Reaudit

- Lower-is-better metrics audited: backend_fail, cost, unnecessary, recover_fail, delay_p50, delay_p95, service_time, service_cost.
- Higher-is-better metrics audited: recall, recall/backend_fail, recall/cost, served_attack_mass, backend_success_attack_mass, served_ratio.
- Gate5 direction issue existed: `True`.
- Rows flagged by Gate6 repair: `43`.
- Metrics flagged by direct CSV scan, if encoded: `[]`; count from encoded flags: `0`.
- Corrected Gate5 status remains: `moderate_success`.
