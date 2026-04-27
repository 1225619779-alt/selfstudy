# Claim Map

| paper location | original claim | support level | safe replacement wording |
|---|---|---|---|
| title | Recovery-aware / consequence-aware larger-system success framing, if any current title implies native case39 scale-up. | weak | <span style="color:red">RED FLAG</span> Avoid 'native case39' or 'scale-up success'. Use: 'False-alarm-aware MTD orchestration with case39 bridge transfer and stress-test evidence'. |
| abstract | Case39 validates the method on a larger native system. | invalid | <span style="color:red">RED FLAG</span> Use: 'On case39, we provide a bridge-transfer stress test with native clean/attack/test artifacts but case14-backed train/val calibration; results expose transfer limitations rather than a native success claim.' |
| contribution 3 | Substantially reduces unnecessary MTD, backend failure burden, defense latency, and cost under the defined operating point. | moderate | Safe for case14. For case39 add: '<span style="color:red">RED FLAG</span> case39 evidence is bridge-transfer/stress-test only, not native larger-system proof.' |
| Table II role column | If Table II lists case39 as native larger-system evidence or scale-up success. | invalid | Use role labels: 'bridge transfer', 'mechanism isolation', 'source-anchored repair attempt', 'native safeguarded negative control', and 'out-of-protocol stress test'. |
| Fig. 4 caption | If Fig. 4 describes case39 as native larger-system evidence. | invalid | <span style="color:red">RED FLAG</span> Use: 'Case39 bridge-transfer stress test under frozen case14 dev selection; lower recall/cost profile indicates transfer limitation.' |
| Fig. 5 caption | If Fig. 5 says mechanism evidence proves scale-up success. | weak | Use: 'Mechanism decomposition of where recall/cost changes arise across replay, anchored retune, safeguarded retune, and unconstrained local retune.' |
| conclusion | The method is validated on a larger native case39 system. | invalid | <span style="color:red">RED FLAG</span> Use: 'The larger-system study is currently evidence of bridge transfer and stress behavior; full native case39 train/val remains feasible but not yet canonical.' |
| limitation paragraph | Case39 limitations are minor or only computational. | weak | <span style="color:red">RED FLAG</span> State explicitly: train/val calibration and winner selection are not fully native, anti-write evidence lacks a current-run STAMP, and severity truth is a proxy. |
| consequence-aware wording | Physical consequence / recovery consequence is fully modeled. | weak | <span style="color:red">RED FLAG</span> Use: 'learned expected-consequence proxy'. The current truth target is still max(ang_no,0)*max(ang_str,0); recover_fail is available but not yet in the scheduler objective. |

## Terms Requiring Replacement

- <span style="color:red">RED FLAG</span> `native case39`: replace with `case39 bridge-transfer stress test` unless train/val and winner selection are made canonical-native.
- <span style="color:red">RED FLAG</span> `larger-system evidence`: replace with `larger-system bridge evidence`.
- <span style="color:red">RED FLAG</span> `scale-up success`: replace with `scale-up stress behavior / transfer limitation`.
- <span style="color:red">RED FLAG</span> `physical consequence` or `recovery consequence`: replace with `expected-consequence proxy` unless a recovery-aware truth label is explicitly used.
