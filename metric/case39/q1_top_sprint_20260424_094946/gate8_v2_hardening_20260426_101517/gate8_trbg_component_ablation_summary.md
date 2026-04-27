# Gate8 TRBG Component Ablation Summary

This ablation is diagnostic only and does not replace locked TRBG-source.

- Full TRBG alpha=1 backend_fail reduction: `0.0849`.
- Fail-only alpha=1 backend_fail reduction: `0.1112`.
- Cost-time-only alpha=1 backend_fail reduction: `-0.0123`.
- Full TRBG better/equivalent vs fail-only under the diagnostic rule: `False`.
- Full TRBG better/equivalent vs cost-time-only under the diagnostic rule: `True`.
- Backend_fail reduction driver: `fail_component`.
- If a simpler component looks better on confirm, it remains appendix observation because Gate8 is not authorized to replace the locked main method.
