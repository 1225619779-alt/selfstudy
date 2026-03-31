# Draft v1 — Recovery-aware Verification-Trigger Policy for False-Alarm-Aware MTD Orchestration

## Candidate titles

1. **A Recovery-Aware Verification-Trigger Policy for Reducing False-Alarm-Induced Unnecessary Moving Target Defense in Power Systems**
2. **From Alarm to Defense: A Recovery-Aware Verification Gate for False-Alarm-Aware MTD Deployment**
3. **Recovery-Aware Verification Before Defense: Reducing False-Alarm-Induced MTD Burden in Cyber-Physical Power Systems**

---

## Abstract (English, submission-oriented)

Data-driven detectors can effectively identify false data injection attacks (FDIAs), but their false positives may unnecessarily trigger expensive backend moving target defense (MTD) actions. This issue is especially critical in hybrid data-and-physics defense pipelines, where a positive alarm is often mechanically upgraded into a backend optimization call. In this work, we revisit the open-source DDET-MTD framework and insert a recovery-aware verification layer between recovery and defense, without modifying either the detector backbone or the MTD optimizer itself. The proposed policy constructs a scalar verification score from the post-recovery phase-correction residual on non-reference buses and triggers backend MTD only when this score exceeds a verification threshold.

The key contribution is not a new detector or a new optimizer, but a decision-layer policy that reduces false-alarm-induced backend burden while preserving the original frontend detection behavior. On 7021 clean samples, the frontend false-alarm rate remains unchanged at 0.167925, while the main operating point (τ = 0.021) reduces the backend deployment rate among alarms from 1.000 to 0.110263 and the unnecessary MTD deployment rate from 0.167925 to 0.018516. At the same time, the backend failure rate per false alarm drops from 0.105174 to 0.038168, and the mean stage-I/stage-II defense time per false alarm decreases from 1.219028/15.945093 to 0.273272/1.627983. A stricter operating point (τ = 0.030) further lowers deployment and cost burdens, yielding a more conservative trade-off. These results show that positive alarms should not be treated as an automatic command for defense deployment, and that a lightweight recovery-aware verification policy can substantially improve the operational efficiency of alarm-driven MTD orchestration.

**Keywords:** false data injection attack, moving target defense, alarm verification, cyber-physical power systems, recovery-aware triggering

---

## 1. Introduction

### 1.1 Motivation

False data injection attacks (FDIAs) remain a major threat to state estimation and control in cyber-physical power systems. Recent defense pipelines increasingly combine data-driven frontends with physics-based backends: a detector first raises an alarm, recovery and state correction are then performed, and a backend MTD optimizer is finally invoked to improve system resilience. This detect–recover–defend pipeline is appealing because it blends fast anomaly screening with physically grounded reconfiguration.

However, once such a pipeline is deployed in practice, a crucial systems-level issue emerges: a positive frontend alarm does not necessarily imply that a backend defense should always be launched. In particular, data-driven detection modules may exhibit non-negligible false positives under clean operation. If every positive alarm is mechanically upgraded to a backend MTD call, the system pays unnecessary optimization, latency, failure, and operational-cost burdens during normal operation.

This issue is not equivalent to improving the detector itself, nor is it the same as redesigning the backend MTD optimizer. Instead, it lies at the interface between alarm generation and defense deployment. Therefore, the main question addressed in this paper is: **when a data-driven detector raises an alarm, should the backend MTD always be triggered?**

### 1.2 Positioning and gap

Our work is built on the open-source DDET-MTD framework. Rather than replacing the detector backbone or rewriting the two-stage MTD optimization, we introduce a lightweight recovery-aware verification-trigger policy between recovery and defense. The contribution is thus a **decision-layer orchestration policy**: the frontend alarm is first verified using a physics-informed signal derived from recovery, and only then is the backend MTD conditionally triggered.

This positioning is important. First, it matches the practical bottleneck of the original pipeline: false alarms create unnecessary backend burden. Second, it yields a tractable and publishable extension for a master-level project because it does not require rebuilding the full learning or optimization stack. Third, it directly responds to a gap that is less crowded than detector-architecture improvements or fully new MTD optimizers.

### 1.3 Main idea and contributions

The proposed policy computes a verification score from the post-recovery phase-correction residual on non-reference buses:

\[
\text{verify\_score} = \| c_{\text{recover,no-ref}} \|_2.
\]

Backend MTD is triggered only if the verification score exceeds a threshold \(\tau\). The main operating point is selected as the clean false-alarm p90 threshold (\(\tau = 0.021\)), while a stricter sensitivity point is selected as the clean false-alarm p95 threshold (\(\tau = 0.030\)).

The contributions of this paper are summarized as follows:

1. We reformulate alarm-driven defense deployment as a **verification-trigger problem** instead of an automatic alarm-to-defense escalation problem.
2. We propose a **recovery-aware verification score** that uses the residual phase-correction intensity after recovery, without modifying either the detector or the backend MTD optimizer.
3. We show on strictly paired clean false alarms that the proposed gate substantially reduces **unnecessary MTD deployment**, **backend failure burden**, **defense latency burden**, and **incremental operating cost burden**, while keeping the frontend false-alarm rate unchanged.
4. We provide a threshold sensitivity analysis with two operating points and a focused matched-budget ablation that clarifies the role of the decision statistic under the same trigger form.

---

## 2. Problem formulation and method

### 2.1 Measurement model and attack model

The standard measurement model is

\[
z = h(x) + e,
\]

where \(z\) denotes the measurement vector, \(x\) the system state, \(h(\cdot)\) the AC power-flow measurement map, and \(e\) the measurement noise. Under a false data injection attack,

\[
z^{\text{att}} = h(x) + a + e,
\]

where \(a\) is the attack vector. In a locally linearized view, stealthy attacks are commonly expressed as

\[
a \approx Hc,
\]

where \(H\) is the Jacobian and \(c\) is a state perturbation vector.

### 2.2 Verification-trigger policy

Let the original DDET-MTD pipeline be

\[
\text{detect} \rightarrow \text{recover} \rightarrow \text{defend}.
\]

In the original alarm-driven setting, every positive detection may be escalated to backend MTD. We instead insert a verification layer:

\[
\text{detect} \rightarrow \text{recover} \rightarrow \text{verify} \rightarrow \text{trigger/skip} \rightarrow \text{defend}.
\]

The recovery module outputs a recovered state. We then build a phase-correction residual vector and remove the reference bus component. The verification score is defined as

\[
\text{verify\_score} = \| c_{\text{recover,no-ref}} \|_2.
\]

The trigger rule is

\[
\text{trigger MTD} \iff \text{verify\_score} \geq \tau,
\]

\[
\text{skip MTD} \iff \text{verify\_score} < \tau.
\]

This gate is placed directly between `recover(...)` and `mtd_optim(...)`.

### 2.3 Threshold calibration

Thresholds are calibrated from the clean false-alarm score distribution. The clean p90 value yields the main operating point \(\tau = 0.021\), and the clean p95 value yields a stricter operating point \(\tau = 0.030\). This makes the threshold choice data-grounded rather than ad hoc.

---

## 3. Experimental setup

### 3.1 Base framework and implementation scope

Experiments are built on the open-source DDET-MTD codebase. The detector backbone and the two-stage MTD optimizer are not modified. Our implementation only changes the decision logic between recovery and defense.

### 3.2 Strict comparability

To ensure paired comparisons, the frontend pipeline is made repeatable by disabling dataset shuffling and using sample-wise fixed attack seeds. Baseline and gated runs are therefore evaluated on the same sample order, the same attack signatures, and the same frontend alarms.

### 3.3 Metrics

For clean false-alarm analysis, we report:

- **Front-end FAR**
- **Backend MTD deployment rate among alarms**
- **Alarm rejection rate**
- **Unnecessary MTD deployment rate**
- **Backend failure rate per false alarm**
- **Mean stage-I defense time per false alarm**
- **Mean stage-II defense time per false alarm**
- **Mean stage-I incremental operating cost per false alarm**
- **Mean stage-II incremental operating cost per false alarm**

For attack-side supporting analysis, we report trigger retention and preservation of stronger attacks.

---

## 4. Results

### 4.1 Clean false-alarm burden reduction (main result)

The main result is obtained on 7021 clean samples. The frontend detector produces 1179 false alarms, corresponding to a frontend FAR of 0.167925. This frontend FAR remains unchanged across the baseline, the main operating point, and the stricter operating point, which is critical because the proposed method does not claim to improve the detector itself.

Under the baseline policy (alarm always triggers backend MTD), the backend deployment rate among alarms is 1.000, the unnecessary MTD deployment rate is 0.167925, the backend failure rate per false alarm is 0.105174, and the mean stage-I/stage-II defense time per false alarm is 1.219028/15.945093. The corresponding mean stage-I/stage-II incremental operating cost per false alarm is 29.604553/4.161843.

At the main operating point \(\tau = 0.021\), the backend deployment rate among alarms decreases to 0.110263 and the alarm rejection rate increases to 0.889737. The unnecessary MTD deployment rate drops to 0.018516, corresponding to an 88.97% reduction relative to the baseline. At the same time, the backend failure rate per false alarm drops to 0.038168, the mean stage-I/stage-II defense time per false alarm drops to 0.273272/1.627983, and the mean stage-I/stage-II incremental operating cost per false alarm drops to 2.109451/0.482992. These correspond to reductions of 63.71%, 77.58%, 89.79%, 92.87%, and 88.39%, respectively.

These results directly support the core claim of this paper: **a positive frontend alarm should not automatically imply backend MTD deployment**. Instead, a recovery-aware verification layer can filter false-alarm-induced backend actions and substantially reduce the resulting operational burden.

### 4.2 Threshold sensitivity: main vs. stricter operating point

A stricter operating point is further obtained by setting \(\tau = 0.030\), corresponding to the clean p95 threshold. At this point, the backend deployment rate among alarms further decreases to 0.076336 and the unnecessary MTD deployment rate further decreases to 0.012819, a 92.37% reduction relative to the baseline. The backend failure rate per false alarm further drops to 0.022901. The mean stage-I/stage-II incremental operating cost per false alarm decreases to 1.658756/0.406886.

However, the stricter operating point does not uniformly dominate the main operating point. In particular, the mean stage-II defense time per false alarm increases from 1.627983 at \(\tau = 0.021\) to 1.944345 at \(\tau = 0.030\). Therefore, \(\tau = 0.030\) should be described as a **stricter trade-off point** rather than a universally better setting.

### 4.3 Focused matched-budget ablation (supporting / appendix)

To address the potential criticism that the proposed method is “just an if-else gate,” we compare two decision statistics under the same trigger form and the same clean deployment budget: the detector reconstruction loss and the proposed recovery-aware physics score.

At the main matched budget (130 selected clean alarms), the detector-loss gate achieves attack retention of 0.922671, while the proposed physics score improves retention to 0.937610. The physics-aware score also substantially reduces stage-I/stage-II incremental operating cost per false alarm from 6.025685/1.237807 to 2.109451/0.482992. However, the detector-loss gate yields lower failure and time burdens at this matched budget.

At the stricter matched budget (90 selected clean alarms), the physics-aware score still produces much lower incremental operating cost than the detector-loss gate, but its attack retention becomes lower and its failure/time burdens remain higher. Therefore, the ablation should not be presented as a uniform dominance result. Instead, it should be used to show that **the decision statistic matters**, and that the proposed recovery-aware score induces a different retention–cost trade-off from a detector-confidence score under the same one-threshold trigger operator.

### 4.4 Attack-side supporting evidence

Attack-side results should be framed as supporting evidence rather than the main selling point. Existing observations show that the gate primarily filters weak or boundary alarms while largely preserving strong-attack alarms. This supports the security plausibility of the trigger policy, but the main value of the paper still lies in clean false-alarm burden reduction.

---

## 5. Discussion

### 5.1 What this paper contributes

This paper should be presented as a **decision-layer contribution**, not as a detector paper and not as a new MTD optimization theory paper. Its value lies in reorganizing the alarm-to-defense interface in a hybrid data-and-physics pipeline.

### 5.2 Why clean experiments are the main evidence

The clean experiment isolates the exact mechanism of false-alarm-induced backend burden. Since the frontend FAR remains unchanged, the observed improvements can be attributed to the verification-trigger policy rather than to detector retraining or threshold manipulation at the detector level.

### 5.3 Limitations

The current evidence is strongest on the IEEE 14-bus setting. The attack-side benefit is mainly retention-based and not a direct improvement in attack-side per-alarm latency. The matched-budget ablation reveals a non-trivial trade-off rather than universal dominance. Finally, a mixed clean–attack timeline case study and a larger test case (e.g., case39) would strengthen external credibility.

---

## 6. Conclusion

This paper revisits alarm-driven MTD deployment from a systems perspective. Instead of automatically escalating every positive alarm into backend defense, we propose a recovery-aware verification-trigger policy that inserts a lightweight decision layer between recovery and MTD optimization. On strictly paired clean false alarms, the method leaves the frontend FAR unchanged while substantially reducing unnecessary backend deployment, backend failure burden, defense latency burden, and incremental operating cost burden. These findings indicate that the efficiency of hybrid cyber-physical defense pipelines can be significantly improved by rethinking the alarm-to-defense interface, even without changing the detector backbone or the MTD optimizer.

---

## What to keep / what not to overclaim

### Safe claims

- The frontend FAR is unchanged.
- The proposed gate substantially reduces false-alarm-induced backend burden on clean paired comparisons.
- The main operating point is \(\tau = 0.021\); \(\tau = 0.030\) is a stricter trade-off point.
- The method is a decision-layer/orchestration-layer contribution.

### Claims to avoid

- “The detector FAR is improved.”
- “The proposed score uniformly dominates all alternatives.”
- “The paper proposes a new MTD optimizer.”
- “The gate itself has directly measured negligible runtime overhead.”

---

## Suggested figure list

1. Framework diagram: detect → recover → verify → trigger/skip → defend
2. Verify-score distribution on clean false alarms vs attack true alarms
3. Main clean burden table (Baseline / τ=0.021 / τ=0.030)
4. Trade-off bar chart for clean burden reductions
5. Focused ablation table or figure (appendix or short正文小节)

---

## Suggested table list

### Table 1. Main clean results

| Policy | τ | Front-end FAR | Trigger count | Alarm rejection rate | Unnecessary MTD deployment rate | Failure / false alarm | Stage-I time / false alarm | Stage-II time / false alarm | Stage-I cost / false alarm | Stage-II cost / false alarm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | -1.0 | 0.167925 | 1179 | 0.000000 | 0.167925 | 0.105174 | 1.219028 | 15.945093 | 29.604553 | 4.161843 |
| Main OP | 0.021 | 0.167925 | 130 | 0.889737 | 0.018516 | 0.038168 | 0.273272 | 1.627983 | 2.109451 | 0.482992 |
| Strict OP | 0.030 | 0.167925 | 90 | 0.923664 | 0.012819 | 0.022901 | 0.235028 | 1.944345 | 1.658756 | 0.406886 |

### Table 2. Matched-budget ablation (supporting)

| Budget | Score | Clean selected alarms | Unnecessary MTD deployment rate | Failure / false alarm | Stage-I time / false alarm | Stage-II time / false alarm | Stage-I cost / false alarm | Stage-II cost / false alarm | Attack retention |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Main OP | detector loss | 130 | 0.018516 | 0.013571 | 0.162294 | 1.854547 | 6.025685 | 1.237807 | 0.922671 |
| Main OP | proposed physics score | 130 | 0.018516 | 0.038168 | 0.275387 | 2.521660 | 2.109451 | 0.482992 | 0.937610 |
| Strict OP | detector loss | 90 | 0.012819 | 0.007634 | 0.093751 | 1.186803 | 5.033306 | 1.201097 | 0.911248 |
| Strict OP | proposed physics score | 90 | 0.012819 | 0.022901 | 0.228268 | 2.233804 | 1.658756 | 0.406886 | 0.855888 |
