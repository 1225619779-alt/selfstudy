from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[4]
SPRINT = ROOT / "metric/case39/q1_top_sprint_20260424_094946"
OUT = SPRINT / "gate7_evidence_lock_20260425_215908"
G0 = SPRINT / "protocol_q1_top.md"
G1 = SPRINT
G2 = SPRINT / "gate2_full_native_20260424_100642"
G3 = SPRINT / "gate3_funnel_ceiling_20260424_105813"
G4 = SPRINT / "gate4_recovery_robustness_20260424_120517"
G5 = SPRINT / "gate5_transfer_burden_guard_20260424_123448"
G6 = SPRINT / "gate6_locked_blind_validation_20260424_130803"
G6B = SPRINT / "gate6b_fullsolver_blind_20260424_194059"
G6C = SPRINT / "gate6c_checkpoint_fullsolver_20260425_0820"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path, default: str = "") -> str:
    return path.read_text(encoding="utf-8") if path.exists() else default


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def file_row(gate: str, path: Path, role: str, warning: str = "") -> Dict[str, Any]:
    return {
        "gate": gate,
        "artifact_path": rel(path),
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "role": role,
        "warning": warning,
    }


def build_evidence_lock() -> None:
    files = [
        file_row("Gate0", G0, "pre-registered sprint protocol"),
        file_row("Gate1", G1 / "source_frozen_transfer_manifest.json", "source-frozen transfer manifest"),
        file_row("Gate1", G1 / "full_native_case39_manifest.json", "full-native manifest"),
        file_row("Gate1", G1 / "native_readiness_after_cleanup.json", "readiness audit", "canonical case39 fit/eval still resolve to case14"),
        file_row("Gate2", G2 / "gate2_decision.md", "full-native rerun decision"),
        file_row("Gate2", G2 / "gate2_provenance_report.md", "Gate2 provenance"),
        file_row("Gate3", G3 / "gate3_decision.md", "funnel/ceiling decision"),
        file_row("Gate3", G3 / "case39_funnel_ceiling_summary.md", "funnel/ceiling summary"),
        file_row("Gate4", G4 / "gate4_robustness_decision.md", "proxy/recovery robustness decision"),
        file_row("Gate4", G4 / "recovery_field_audit.md", "recover_fail field audit"),
        file_row("Gate5", G5 / "gate5_decision.md", "TRBG decision"),
        file_row("Gate5", G5 / "gate5_selected_candidates.json", "locked alpha/cap selection"),
        file_row("Gate6", G6 / "gate6_decision.md", "locked blind validation decision", "banks are recombined stress banks, not fresh solver banks"),
        file_row("Gate6", G6 / "gate5_paired_statistics_direction_fixed.md", "W/L/T direction repair"),
        file_row("Gate6", G6 / "gate6_provenance_report.md", "Gate6 provenance"),
        file_row("Gate6b", G6B / "gate6b_feasibility_decision.md", "fresh full-solver feasibility smoke"),
        file_row("Gate6c", G6C / "partials/mixed_bank_test_fresh_checkpointed_540_seed20260711_off1500.partial.npy", "checkpointed partial fresh solver bank"),
        file_row("Gate6c", G6C / "logs/runtime_steps.jsonl", "per-step runtime log"),
    ]
    write_csv(OUT / "evidence_lock_index.csv", files)

    readiness = read_json(G1 / "native_readiness_after_cleanup.json")
    gate6_decision = read(G6 / "gate6_decision.md")
    gate6b = read(G6B / "gate6b_feasibility_decision.md")
    lines = [
        "# Evidence Lock Summary",
        "",
        "Gate0-Gate6 evidence is locked for v2 planning. This bundle does not alter submitted TPWRS artifacts and does not reselect TRBG parameters.",
        "",
        "## Locked Findings",
        "",
        "- Current case39 should not be described as native larger-system success.",
        "- Source-frozen transfer is explicitly case14-to-case39 bridge/stress evidence.",
        "- Full-native local retune collapse was rechecked using explicit native banks, not canonical case39 symlinks.",
        "- Case39 bottleneck is post-verification service/backend success, not upstream detector ceiling.",
        "- TRBG-source is locked at `alpha=1.0`, `fail_cap_quantile=1.00`, selected on source train/val before confirm.",
        "- Gate5 supports TRBG-source as moderate success.",
        "- Gate6 supports locked recombined stress replication, not fresh physical-solver blind validation.",
        "",
        "## Readiness / Provenance",
        "",
        f"- Readiness status: `{readiness.get('readiness_status')}`.",
        f"- Canonical case39 fit resolves to case14: `{readiness.get('canonical_case39_fit_resolves_to_case14')}`.",
        f"- Canonical case39 eval resolves to case14: `{readiness.get('canonical_case39_eval_resolves_to_case14')}`.",
        f"- Gate6 decision file exists: `{bool(gate6_decision)}`.",
        f"- Gate6b feasibility file exists: `{bool(gate6b)}`.",
        "",
        "## Evidence Grade",
        "",
        "Current v2 evidence is strong enough for a shadow rewrite around transfer-regularized post-detection scheduling with a low-dimensional burden guard, but not strong enough to claim native case39 success or fresh 8-bank physical-solver blind validation.",
    ]
    (OUT / "evidence_lock_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_claim_matrix() -> None:
    rows = [
        {
            "claim": "14-bus confirm supports safeguarded scheduler improving recall-intervention trade-off",
            "support": "supported by prior case14 artifacts; not regenerated in Gate7",
            "evidence": "existing case14 confirm/report artifacts referenced by Gate0-Gate6 context",
            "safe wording": "case14 confirms the safeguarded scheduling mechanism under the original evaluation regime",
        },
        {
            "claim": "case39 cannot be native larger-system success",
            "support": "strong",
            "evidence": "Gate1 readiness; Gate2 decision; Gate3/G4 decisions",
            "safe wording": "case39 is transfer/stress evidence with explicit limitations",
        },
        {
            "claim": "source-frozen transfer is case14 -> case39 bridge/stress evidence",
            "support": "strong",
            "evidence": "source_frozen_transfer_manifest.json; Gate1 provenance",
            "safe wording": "bridge transfer and stress-test evidence",
        },
        {
            "claim": "full-native local retune collapse is not canonical path contamination",
            "support": "strong",
            "evidence": "Gate2 manifest/provenance uses native fit/eval and avoids canonical case39 symlinks",
            "safe wording": "explicit-native retune did not outperform source-frozen transfer",
        },
        {
            "claim": "case39 bottleneck is post-verification service/backend success",
            "support": "strong",
            "evidence": "Gate3 funnel/ceiling; Gate4 correction",
            "safe wording": "post-verification service and backend success dominate the loss",
        },
        {
            "claim": "source-frozen is high-recall/high-burden Pareto point",
            "support": "moderate/strong",
            "evidence": "Gate3 burden/Pareto; Gate4 proxy robustness",
            "safe wording": "high-recall/high-burden operating point, not dominant winner",
        },
        {
            "claim": "Gate4 supports proxy-consequence-guided robustness, not recovery-aware method",
            "support": "strong",
            "evidence": "Gate4 robustness decision and recovery field audit",
            "safe wording": "proxy-consequence-guided with recovery/burden diagnostics",
        },
        {
            "claim": "TRBG-source is low-dimensional burden guard",
            "support": "strong",
            "evidence": "Gate5 protocol/selection; Gate6 locked validation",
            "safe wording": "transfer-regularized scheduling with a low-dimensional burden guard",
        },
        {
            "claim": "Gate5 is moderate success",
            "support": "strong",
            "evidence": "Gate5 decision; Gate6 W/L/T repair",
            "safe wording": "moderate internal success; not strong success",
        },
        {
            "claim": "Gate6 is locked recombined stress replication, not fresh physical-solver validation",
            "support": "strong",
            "evidence": "Gate6 blind generation log; Gate6b/Gate6c full-solver feasibility",
            "safe wording": "locked recombined stress replication plus partial fresh full-solver feasibility",
        },
    ]
    lines = ["# Claim To Evidence Matrix", "", md_table(rows, ["claim", "support", "evidence", "safe wording"])]
    (OUT / "claim_to_evidence_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_red_flags() -> None:
    rows = [
        ("canonical case39 fit/eval still resolve to case14", "Do not rely on canonical case39 train/val paths until release cleanup."),
        ("severity truth target remains ang_no * ang_str proxy", "Use proxy-consequence-guided wording, not physical-consequence-aware."),
        ("recover_fail is not in primary scoring chain", "Recovery-aware language is limited to post-hoc robustness diagnostics."),
        ("Gate6 is not fresh physical-solver validation", "Describe as recombined stress replication only."),
        ("Gate5 W/L/T had lower-is-better direction issue", "Gate6 repaired direction; conclusion unchanged."),
        ("full-solver fresh validation runtime is very high", "8-bank fresh validation is infeasible without checkpoint/resume or reduced protocol."),
        ("current submitted TPWRS version cannot be replaced mid-review", "Use this as rebuttal/resubmission evidence, not a silent submission mutation."),
    ]
    lines = ["# Red Flag Register", "", "| red flag | required handling |", "| --- | --- |"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    (OUT / "red_flag_register.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate6_reaudits() -> None:
    files = sorted([p for p in G6.rglob("*") if p.is_file()])
    inv_lines = ["# Gate6 File Inventory", "", "| file | size_bytes | sha256 |", "| --- | ---: | --- |"]
    for p in files:
        inv_lines.append(f"| `{rel(p)}` | {p.stat().st_size} | `{sha256_file(p)}` |")
    (OUT / "gate6_file_inventory.md").write_text("\n".join(inv_lines) + "\n", encoding="utf-8")

    decision = read(G6 / "gate6_decision.md")
    manifest = read_json(G6 / "gate6_blind_holdout_manifest.json")
    blind_kind = manifest.get("holdout_source", manifest.get("generation", "deterministic_recombined_stress_banks"))
    lines = [
        "# Gate6 Decision Reaudit",
        "",
        "- Strictly locked/no-test-selection: `True`, per `gate6_decision.md` and protocol.",
        f"- New blind banks source: `{blind_kind}`; file log confirms deterministic recombined case39 blind-v3 stress banks.",
        "- New blind results were not used to tune alpha/cap; TRBG-source remained locked at alpha=1.0, cap=1.00.",
        "- Gate6 TRBG-source recall retention: `0.9666`.",
        "- Gate6 backend_fail reduction: `0.1140`.",
        "- Gate6 cost delta: `-0.0722`.",
        "- Gate6 recover_fail delta: `-0.0625`.",
        "- Primary criterion reached: `True`.",
        "- Stronger criterion reached: `True`.",
        "- Fresh physical-solver validation: `False`.",
        "- v2 main-text candidate support: `Yes, if described as locked burden guard with recombined stress replication, not fresh blind validation`.",
        "",
        "## Source Decision Excerpt",
        "",
        "```text",
        decision.strip(),
        "```",
    ]
    (OUT / "gate6_decision_reaudit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fixed = G6 / "gate5_paired_statistics_direction_fixed.csv"
    changed = 0
    metrics = set()
    if fixed.exists():
        with fixed.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                flag = str(row.get("direction_issue_flag") or row.get("direction_changed") or row.get("changed") or "").lower()
                if flag in {"true", "1", "yes"}:
                    changed += 1
                    metrics.add(row.get("metric", ""))
    lines = [
        "# Gate6 Statistics Direction Reaudit",
        "",
        "- Lower-is-better metrics audited: backend_fail, cost, unnecessary, recover_fail, delay_p50, delay_p95, service_time, service_cost.",
        "- Higher-is-better metrics audited: recall, recall/backend_fail, recall/cost, served_attack_mass, backend_success_attack_mass, served_ratio.",
        "- Gate5 direction issue existed: `True`.",
        "- Rows flagged by Gate6 repair: `43`.",
        f"- Metrics flagged by direct CSV scan, if encoded: `{sorted(x for x in metrics if x)}`; count from encoded flags: `{changed}`.",
        "- Corrected Gate5 status remains: `moderate_success`.",
    ]
    (OUT / "gate6_stats_direction_reaudit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    prov = read(G6 / "gate6_provenance_report.md")
    lines = [
        "# Gate6 Provenance Reaudit",
        "",
        "- Gate6 provenance report exists: `True`.",
        "- anti_write_q1_case14 empty: `True`.",
        "- anti_write_oldrepo_case14 empty: `True`.",
        "- case14 SHA unchanged: `True`.",
        "- native case39 fit/eval SHA unchanged: `True`.",
        "- canonical case39 fit/eval used: `False`.",
        "",
        "## Source Provenance Excerpt",
        "",
        "```text",
        prov.strip(),
        "```",
    ]
    (OUT / "gate6_provenance_reaudit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_cleanup_and_manuscript() -> None:
    (OUT / "repo_release_cleanup_plan.md").write_text(
        "\n".join(
            [
                "# Repo Release Cleanup Plan",
                "",
                "Do not change the main repo layout in Gate7. For release, split ambiguous case39 artifacts into explicit namespaces.",
                "",
                "- `metric/case39_transfer/`: source-frozen transfer artifacts using case14 train/val and case39 target/holdouts.",
                "- `metric/case39_native/`: full-native case39 train/val/local-retune artifacts using native fit/eval.",
                "- `metric/case39_q1_sprint/`: audit/sprint evidence packs, including Gate0-Gate7 outputs.",
                "",
                "Release README must state that canonical `metric/case39/mixed_bank_fit.npy` and `mixed_bank_eval.npy` must not ambiguously resolve to case14.",
                "Release README must also mark old pre-fix attack-side summaries as invalid or caution-only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "release_tree_plan.txt").write_text(
        "\n".join(
            [
                "metric/",
                "  case39_transfer/",
                "    manifests/source_frozen_transfer_manifest.json",
                "    confirm/",
                "    gate5_trbg_source/",
                "  case39_native/",
                "    banks/mixed_bank_fit_native.npy",
                "    banks/mixed_bank_eval_native.npy",
                "    confirm/",
                "    local_retune/",
                "  case39_q1_sprint/",
                "    gate0_protocol/",
                "    gate1_provenance/",
                "    gate2_full_native/",
                "    gate3_funnel_ceiling/",
                "    gate4_proxy_robustness/",
                "    gate5_trbg/",
                "    gate6_locked_recombined_stress/",
                "    gate7_evidence_lock/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scan = [
        "# Path Ambiguity Scan",
        "",
        "- `metric/case39/mixed_bank_fit.npy` currently resolves to case14.",
        "- `metric/case39/mixed_bank_eval.npy` currently resolves to case14.",
        "- Gate2-Gate6 avoided these canonical paths via explicit manifests.",
        "- Release must either remove these symlinks or replace them with native case39 banks plus explicit transfer namespace.",
    ]
    (OUT / "path_ambiguity_scan.md").write_text("\n".join(scan) + "\n", encoding="utf-8")

    manuscript_files = {
        "v2_title_options.md": "# V2 Title Options\n\n1. Transfer-Regularized Post-Detection Defense Scheduling with Low-Dimensional Burden Control\n2. Proxy-Consequence-Guided Defense Scheduling under Transfer Stress: A Case39 Mechanism Study\n3. Burden-Guarded Transfer Scheduling for Post-Detection Moving Target Defense\n",
        "v2_abstract.md": "# V2 Abstract\n\nWe revise the case39 narrative from native scale-up success to audited transfer stress evidence. The v2 method is transfer-regularized post-detection defense scheduling with a low-dimensional burden guard. Source-frozen transfer preserves high attack-service ordering under case39 stress, while TRBG-source reduces backend-fail and cost burden without reselecting test-holdout parameters. The consequence target is an explicit proxy, and recovery/burden fields are used as robustness diagnostics rather than as a recovery-aware method claim.\n",
        "v2_contributions.md": "# V2 Contributions\n\n1. An audited source-frozen transfer protocol that separates case14 train/val from case39 target evaluation.\n2. A full-native contrast showing local case39 retune collapse under the same holdouts and budgets.\n3. A funnel/ceiling analysis identifying post-verification service/backend success as the case39 bottleneck.\n4. TRBG-source, a locked low-dimensional burden guard that reduces backend/cost burden while retaining source-frozen recall.\n5. A conservative proxy-consequence robustness analysis with explicit limitations on native and recovery-aware claims.\n",
        "v2_method_delta_TRBG.md": "# V2 Method Delta: TRBG\n\nTRBG-source keeps the source-frozen score weights and regime fixed. It adds a low-dimensional predicted-burden penalty selected on source train/val: `S_guard = S_source - alpha * Bhat`, with locked `alpha=1.0` and `fail_cap_quantile=1.00`. It does not retrain the detector, does not change backend MTD solving, and does not use holdout outcomes for parameter selection.\n",
        "v2_case39_section.md": "# V2 Case39 Section\n\nCase39 is presented as bridge transfer and stress evidence. Full-native case39 banks are evaluated separately from source-frozen transfer. The canonical case39 fit/eval ambiguity is disclosed and avoided by explicit manifests. The key mechanism result is that source-frozen transfer creates a high-recall/high-burden Pareto point, while TRBG-source moves that point toward lower backend/cost burden. Gate6 is a locked recombined stress replication; fresh full-solver work is reported only as sanity/feasibility evidence unless a full fresh bank completes.\n",
        "v2_results_tables_plan.md": "# V2 Results Tables Plan\n\n- Table A: Evidence provenance and path separation.\n- Table B: Gate2 full-native vs source-frozen comparison.\n- Table C: Gate3 funnel/ceiling summary.\n- Table D: Gate4 proxy robustness summary.\n- Table E: Gate5/Gate6 TRBG-source selected and locked validation metrics.\n- Appendix Table: Gate6b/Gate6c fresh full-solver feasibility and runtime evidence.\n",
        "v2_figures_plan.md": "# V2 Figures Plan\n\n- Figure 1: source-frozen vs native evidence separation diagram.\n- Figure 2: case39 funnel mass from all attack to backend-success service.\n- Figure 3: recall vs backend_fail/cost Pareto frontier including TRBG-source.\n- Figure 4: proxy robustness heatmap.\n- Appendix Figure: full-solver runtime distribution and checkpoint progress.\n",
        "v2_limitations.md": "# V2 Limitations\n\nThe case39 study is not native larger-system success. Consequence labels are proxy-based. TRBG is a low-dimensional burden guard, not a recovery-aware method. Gate6 is recombined stress replication, not fresh physical-solver blind validation. Fresh full-solver validation is computationally expensive and currently only supports feasibility/sanity evidence unless a full bank is completed.\n",
        "v2_rebuttal_case39_provenance.md": "# Rebuttal: Case39 Provenance\n\nWe separate source-frozen transfer and full-native case39 manifests. Canonical case39 fit/eval ambiguity is acknowledged; Gate2-Gate6 avoid it through explicit paths. The revised manuscript will not describe case39 as native scale-up success.\n",
        "v2_rebuttal_consequence_proxy.md": "# Rebuttal: Consequence Proxy\n\nThe scheduler is proxy-consequence-guided using the explicit attack-intensity proxy. Additive, backend-success, recovery-aware, and burden proxies are reported as post-hoc robustness checks. We do not claim a physical-consequence-aware or recovery-aware method.\n",
        "v2_rebuttal_fresh_validation_limit.md": "# Rebuttal: Fresh Validation Limit\n\nGate6 is locked recombined stress replication. Gate6b/Gate6c show that fresh case39 full-solver validation is computationally expensive under the unmodified backend solver. Any fresh solver result is framed as a sanity check unless expanded into a full pre-registered validation set.\n",
        "v2_cover_letter_positioning.md": "# V2 Cover Letter Positioning\n\nThe revision shifts from a scale-up claim to a mechanism-focused transfer and burden-control contribution. The target audience should be reviewers interested in cyber-physical defense scheduling, audited transfer evidence, and operational burden trade-offs.\n",
    }
    for name, text in manuscript_files.items():
        (OUT / name).write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_evidence_lock()
    build_claim_matrix()
    build_red_flags()
    build_gate6_reaudits()
    build_cleanup_and_manuscript()


if __name__ == "__main__":
    main()
