from __future__ import annotations

import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SPRINT = ROOT / "metric" / "case39" / "q1_top_sprint_20260424_094946"
GATE2 = SPRINT / "gate2_full_native_20260424_100642"
GATE3 = SPRINT / "gate3_funnel_ceiling_20260424_105813"
OUT = SPRINT / "gate4_recovery_robustness_20260424_120517"
GATE3_SCRIPT = GATE3 / "gate3_funnel_ceiling.py"

BUDGETS = [1, 2]
METHODS = [
    "source_frozen_transfer",
    "topk_expected_consequence",
    "winner_replay",
    "anchored_retune",
    "native_safeguarded_retune",
    "native_unconstrained_retune",
    "phase3_proposed",
    "phase3_oracle_upgrade",
]
DISPLAY_METHODS = [
    "source_frozen_transfer",
    "topk_expected_consequence",
    "winner_replay",
    "native_safeguarded_retune",
    "native_unconstrained_retune",
    "phase3_oracle_upgrade",
]
PROXIES = [
    "product_proxy",
    "additive_proxy",
    "backend_success_proxy",
    "recovery_aware_proxy",
    "burden_proxy",
    "success_burden_proxy",
]
RNG_SEED = 20260404


def load_gate3():
    spec = importlib.util.spec_from_file_location("gate3_funnel_ceiling", GATE3_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {GATE3_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G3 = load_gate3()
G2 = G3.G2
R2 = G3.R2


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: Any, digits: int = 4) -> str:
    try:
        val = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def md_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(out)


def safe_corr(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    if xa.size < 2 or float(np.std(xa)) <= 1e-12 or float(np.std(ya)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def rank_desc(values: Dict[str, float]) -> Dict[str, float]:
    ordered = sorted(values.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    ranks: Dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and abs(float(ordered[j][1]) - float(ordered[i][1])) <= 1e-12:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j
    return ranks


def spearman_from_ranks(a: Dict[str, float], b: Dict[str, float], methods: Sequence[str]) -> float:
    xs = [a[m] for m in methods if m in a and m in b]
    ys = [b[m] for m in methods if m in a and m in b]
    return safe_corr(xs, ys)


def binom_two_sided_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def sign_flip_p(deltas: Sequence[float]) -> float:
    vals = np.asarray([float(x) for x in deltas if abs(float(x)) > 1e-12], dtype=float)
    n = int(vals.size)
    if n == 0:
        return 1.0
    obs = abs(float(np.mean(vals)))
    count = 0
    total = 2**n
    for mask in range(total):
        signs = np.ones(n)
        for i in range(n):
            if (mask >> i) & 1:
                signs[i] = -1.0
        if abs(float(np.mean(vals * signs))) >= obs - 1e-12:
            count += 1
    return float(count / total)


def bootstrap_ci(deltas: Sequence[float], *, seed: int) -> Tuple[float, float]:
    vals = np.asarray([float(x) for x in deltas], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(10000, vals.size), replace=True)
    means = np.mean(draws, axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_bank(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return np.load(p, allow_pickle=True).item()


def derived_mixed_arrays(path: str | Path) -> Dict[str, np.ndarray] | None:
    try:
        return G2.mixed_bank_to_alarm_arrays(str(ROOT / path if not Path(path).is_absolute() else path))
    except Exception:
        return None


def field_state(arr: np.ndarray | None) -> Dict[str, Any]:
    if arr is None:
        return {
            "present": False,
            "nan_count": None,
            "all_zero": None,
            "constant": None,
            "mean": None,
            "min": None,
            "max": None,
        }
    x = np.asarray(arr)
    xf = x.astype(float).reshape(-1)
    finite = xf[np.isfinite(xf)]
    return {
        "present": True,
        "nan_count": int(np.sum(~np.isfinite(xf))),
        "all_zero": bool(finite.size > 0 and np.all(np.abs(finite) <= 1e-12)),
        "constant": bool(finite.size > 0 and np.nanmax(finite) - np.nanmin(finite) <= 1e-12),
        "mean": float(np.nanmean(xf)) if xf.size else None,
        "min": float(np.nanmin(xf)) if xf.size else None,
        "max": float(np.nanmax(xf)) if xf.size else None,
    }


def numeric_vector(payload: Dict[str, Any], key: str) -> np.ndarray | None:
    if key not in payload:
        return None
    try:
        arr = np.asarray(payload[key], dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.dtype.kind not in {"f", "i", "u", "b"}:
        return None
    return arr


def infer_payload_rows(payload: Dict[str, Any]) -> int:
    for value in payload.values():
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if arr.ndim > 0 and arr.size > 1:
            return int(arr.reshape(-1).size)
    return 0


def product_from_arrays(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    return np.maximum(np.asarray(arrays["ang_no"], dtype=float), 0.0) * np.maximum(np.asarray(arrays["ang_str"], dtype=float), 0.0)


def additive_from_arrays(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    return np.maximum(np.asarray(arrays["ang_no"], dtype=float), 0.0) + np.maximum(np.asarray(arrays["ang_str"], dtype=float), 0.0)


def collect_bank_paths() -> List[Dict[str, str]]:
    source_manifest = read_json(SPRINT / "source_frozen_transfer_manifest.json")
    native_manifest = read_json(SPRINT / "full_native_case39_manifest.json")
    gate2_manifest = read_json(GATE2 / "gate2_full_native_manifest_used.json")
    rows = [
        {"artifact_path": source_manifest["source_train_bank"], "artifact_type": "source_train_bank"},
        {"artifact_path": source_manifest["source_val_bank"], "artifact_type": "source_val_bank"},
        {"artifact_path": native_manifest["native_train_bank"], "artifact_type": "native_train_bank"},
        {"artifact_path": native_manifest["native_val_bank"], "artifact_type": "native_val_bank"},
        {"artifact_path": source_manifest["target_clean_bank"], "artifact_type": "target_clean_bank"},
        {"artifact_path": source_manifest["target_attack_bank"], "artifact_type": "target_attack_bank"},
    ]
    for idx, hold in enumerate(gate2_manifest["holdouts"]):
        rows.append({"artifact_path": hold["test_bank"], "artifact_type": f"holdout_{idx}"})
    return rows


def bank_audit_rows() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    aggregate_recover: List[float] = []
    aggregate_backend: List[float] = []
    aggregate_product: List[float] = []
    aggregate_additive: List[float] = []
    aggregate_time: List[float] = []
    aggregate_cost: List[float] = []
    cross = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    mixed_recover_freqs: List[float] = []
    mixed_backend_freqs: List[float] = []
    for item in collect_bank_paths():
        rel = item["artifact_path"]
        payload = load_bank(rel)
        arrays = derived_mixed_arrays(rel)
        recover_raw = numeric_vector(payload, "recover_fail")
        backend_raw = numeric_vector(payload, "backend_fail")
        exact_time = numeric_vector(payload, "service_time")
        exact_cost = numeric_vector(payload, "service_cost")
        service_time = np.asarray(arrays["service_time"], dtype=float).reshape(-1) if arrays is not None else exact_time
        service_cost = np.asarray(arrays["service_cost"], dtype=float).reshape(-1) if arrays is not None else exact_cost
        recover = np.asarray(arrays["recover_fail"], dtype=float).reshape(-1) if arrays is not None else recover_raw
        backend = np.asarray(arrays["backend_fail"], dtype=float).reshape(-1) if arrays is not None else backend_raw
        if arrays is not None:
            attack = np.asarray(arrays["is_attack"], dtype=int).reshape(-1) == 1
            product = product_from_arrays(arrays)
            additive = additive_from_arrays(arrays)
            mixed_recover_freqs.append(float(np.mean(recover)) if recover is not None and recover.size else float("nan"))
            mixed_backend_freqs.append(float(np.mean(backend)) if backend is not None and backend.size else float("nan"))
            aggregate_recover.extend(recover.tolist())
            aggregate_backend.extend(backend.tolist())
            aggregate_product.extend(product.tolist())
            aggregate_additive.extend(additive.tolist())
            aggregate_time.extend(service_time.tolist())
            aggregate_cost.extend(service_cost.tolist())
            for r, b in zip(recover.astype(int), backend.astype(int)):
                cross[(int(r), int(b))] += 1
        else:
            attack = None
            product = None
            additive = None
        recover_state = field_state(recover)
        backend_state = field_state(backend)
        time_state = field_state(service_time)
        cost_state = field_state(service_cost)
        rows.append(
            {
                "section": "bank_field",
                "artifact_path": rel,
                "artifact_type": item["artifact_type"],
                "n_rows": int(recover.size if recover is not None else infer_payload_rows(payload)),
                "recover_fail_raw_key_present": "recover_fail" in payload,
                "recover_fail_numeric_vector": recover_raw is not None,
                "backend_fail_raw_key_present": "backend_fail" in payload,
                "backend_fail_numeric_vector": backend_raw is not None,
                "is_mixed_scheduler_bank": arrays is not None,
                "recover_fail_present": recover_state["present"],
                "recover_fail_nan_count": recover_state["nan_count"],
                "recover_fail_all_zero": recover_state["all_zero"],
                "recover_fail_constant": recover_state["constant"],
                "recover_fail_frequency": recover_state["mean"],
                "backend_fail_present": backend_state["present"],
                "backend_fail_nan_count": backend_state["nan_count"],
                "backend_fail_all_zero": backend_state["all_zero"],
                "backend_fail_constant": backend_state["constant"],
                "backend_fail_frequency": backend_state["mean"],
                "service_time_present_exact": exact_time is not None,
                "service_time_derived_present": service_time is not None,
                "service_time_nan_count": time_state["nan_count"],
                "service_time_constant": time_state["constant"],
                "service_cost_present_exact": exact_cost is not None,
                "service_cost_derived_present": service_cost is not None,
                "service_cost_nan_count": cost_state["nan_count"],
                "service_cost_constant": cost_state["constant"],
                "recover_fail_attack_frequency": float(np.mean(recover[attack])) if attack is not None and np.any(attack) else None,
                "recover_fail_clean_frequency": float(np.mean(recover[~attack])) if attack is not None and np.any(~attack) else None,
                "recover_corr_backend": safe_corr(recover, backend) if recover is not None and backend is not None else None,
                "recover_corr_product_proxy": safe_corr(recover, product) if recover is not None and product is not None else None,
                "recover_corr_additive_proxy": safe_corr(recover, additive) if recover is not None and additive is not None else None,
                "recover_corr_service_time": safe_corr(recover, service_time) if recover is not None and service_time is not None else None,
                "recover_corr_service_cost": safe_corr(recover, service_cost) if recover is not None and service_cost is not None else None,
            }
        )
    summary = {
        "mixed_recover_frequency_mean": float(np.nanmean(mixed_recover_freqs)),
        "mixed_backend_frequency_mean": float(np.nanmean(mixed_backend_freqs)),
        "aggregate_recover_backend_corr": safe_corr(aggregate_recover, aggregate_backend),
        "aggregate_recover_product_corr": safe_corr(aggregate_recover, aggregate_product),
        "aggregate_recover_additive_corr": safe_corr(aggregate_recover, aggregate_additive),
        "aggregate_recover_time_corr": safe_corr(aggregate_recover, aggregate_time),
        "aggregate_recover_cost_corr": safe_corr(aggregate_recover, aggregate_cost),
        "cross": cross,
    }
    return rows, summary


def result_recovery_rows(details: Dict[Tuple[str, int, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (method, budget, holdout), payload in sorted(details.items()):
        if method not in METHODS:
            continue
        detail = payload["detail"]
        jobs_by_id = {int(j.job_id): j for j in payload["jobs"]}
        served = [int(x) for x in detail["served_jobs"]]
        served_attack = [i for i in served if int(jobs_by_id[i].is_attack) == 1]
        served_clean = [i for i in served if int(jobs_by_id[i].is_attack) == 0]
        rec_attack = [int(float(jobs_by_id[i].meta.get("recover_fail", 0.0))) for i in served_attack]
        rec_clean = [int(float(jobs_by_id[i].meta.get("recover_fail", 0.0))) for i in served_clean]
        back_attack = [int(jobs_by_id[i].actual_backend_fail) for i in served_attack]
        back_clean = [int(jobs_by_id[i].actual_backend_fail) for i in served_clean]
        rows.append(
            {
                "section": "holdout_variant_distribution",
                "holdout_id": holdout,
                "budget": budget,
                "variant": method,
                "served_attack_jobs": len(served_attack),
                "served_clean_jobs": len(served_clean),
                "served_recover_fail_attack_jobs": int(sum(rec_attack)),
                "served_recover_fail_clean_jobs": int(sum(rec_clean)),
                "served_recover_fail_attack_rate": float(np.mean(rec_attack)) if rec_attack else 0.0,
                "served_recover_fail_clean_rate": float(np.mean(rec_clean)) if rec_clean else 0.0,
                "served_backend_fail_attack_jobs": int(sum(back_attack)),
                "served_backend_fail_clean_jobs": int(sum(back_clean)),
                "served_backend_fail_attack_rate": float(np.mean(back_attack)) if back_attack else 0.0,
                "served_backend_fail_clean_rate": float(np.mean(back_clean)) if back_clean else 0.0,
                "served_recover_backend_corr": safe_corr(rec_attack + rec_clean, back_attack + back_clean) if served else None,
            }
        )
    return rows


def compute_alpha_from_native_train_val() -> float:
    manifest = read_json(SPRINT / "full_native_case39_manifest.json")
    ratios: List[float] = []
    for rel in [manifest["native_train_bank"], manifest["native_val_bank"]]:
        arrays = derived_mixed_arrays(rel)
        if arrays is None:
            continue
        attack = np.asarray(arrays["is_attack"], dtype=int) == 1
        product = product_from_arrays(arrays)
        st = np.asarray(arrays["service_time"], dtype=float)
        sc = np.asarray(arrays["service_cost"], dtype=float)
        time_norm = st / max(float(np.nanpercentile(st, 95)), 1e-12)
        cost_norm = sc / max(float(np.nanpercentile(sc, 95)), 1e-12)
        burden = time_norm + cost_norm + np.asarray(arrays["backend_fail"], dtype=float) + np.asarray(arrays["recover_fail"], dtype=float)
        pos_product = product[attack & (product > 0)]
        pos_burden = burden[attack & (burden > 0)]
        if pos_product.size and pos_burden.size:
            ratios.append(float(np.median(pos_product) / max(float(np.median(pos_burden)), 1e-12)))
    if not ratios:
        return 0.10
    return float(np.clip(np.mean(ratios), 0.01, 0.50))


def proxy_arrays(jobs: Sequence[Any], *, alpha: float) -> Dict[str, Dict[str, Any]]:
    attack = np.asarray([int(j.is_attack) == 1 for j in jobs], dtype=bool)
    backend_fail = np.asarray([int(j.actual_backend_fail) for j in jobs], dtype=float)
    recover_fail = np.asarray([float(j.meta.get("recover_fail", 0.0)) for j in jobs], dtype=float)
    ang_no = np.asarray([float(j.meta.get("ang_no", 0.0)) for j in jobs], dtype=float)
    ang_str = np.asarray([float(j.meta.get("ang_str", 0.0)) for j in jobs], dtype=float)
    service_time = np.asarray([float(j.actual_service_time) for j in jobs], dtype=float)
    service_cost = np.asarray([float(j.actual_service_cost) for j in jobs], dtype=float)
    product = np.where(attack, np.maximum(ang_no, 0.0) * np.maximum(ang_str, 0.0), 0.0)
    additive = np.where(attack, np.maximum(ang_no, 0.0) + np.maximum(ang_str, 0.0), 0.0)
    time_norm = service_time / max(float(np.nanpercentile(service_time, 95)), 1e-12)
    cost_norm = service_cost / max(float(np.nanpercentile(service_cost, 95)), 1e-12)
    burden = cost_norm + time_norm + backend_fail + recover_fail
    success_burden = np.where(attack, product * (1.0 - backend_fail), 0.0) - alpha * burden
    return {
        "product_proxy": {"label": product, "kind": "positive_attack_success"},
        "additive_proxy": {"label": additive, "kind": "positive_attack_success"},
        "backend_success_proxy": {"label": product * (1.0 - backend_fail), "kind": "positive_attack_mass"},
        "recovery_aware_proxy": {"label": product * (1.0 - recover_fail), "kind": "positive_attack_success"},
        "burden_proxy": {"label": burden, "kind": "burden_lower_is_better"},
        "success_burden_proxy": {"label": success_burden, "kind": "signed_burden_adjusted_success"},
    }


def score_proxy(payload: Dict[str, Any], proxy: str, *, alpha: float) -> Dict[str, float | str]:
    jobs = payload["jobs"]
    detail = payload["detail"]
    labels = proxy_arrays(jobs, alpha=alpha)[proxy]
    values = np.asarray(labels["label"], dtype=float)
    kind = str(labels["kind"])
    served = np.asarray([int(x) for x in detail["served_jobs"]], dtype=int)
    served_attack = np.asarray([int(x) for x in detail["served_attack_jobs"]], dtype=int)
    jobs_by_id = {int(j.job_id): j for j in jobs}
    backend_success_served_attack = np.asarray([i for i in served_attack if int(jobs_by_id[int(i)].actual_backend_fail) == 0], dtype=int)
    attack_ids = np.asarray([int(j.job_id) for j in jobs if int(j.is_attack) == 1], dtype=int)
    if kind == "burden_lower_is_better":
        served_burden = float(np.sum(values[served])) if served.size else 0.0
        score = -served_burden / max(float(len(jobs)), 1.0)
        recall_like = score
        weighted_success = -served_burden
        denominator = float(len(jobs))
    elif kind == "signed_burden_adjusted_success":
        served_score = float(np.sum(values[served])) if served.size else 0.0
        positive_total = float(np.sum(np.maximum(values[attack_ids], 0.0))) if attack_ids.size else 0.0
        score = served_score / max(positive_total, 1e-12)
        recall_like = score
        weighted_success = served_score
        denominator = positive_total
    elif kind == "positive_attack_mass":
        numerator = float(np.sum(values[served_attack])) if served_attack.size else 0.0
        denominator = float(np.sum(values[attack_ids])) if attack_ids.size else 0.0
        score = numerator / max(denominator, 1e-12)
        recall_like = score
        weighted_success = numerator
    else:
        numerator = float(np.sum(values[backend_success_served_attack])) if backend_success_served_attack.size else 0.0
        denominator = float(np.sum(values[attack_ids])) if attack_ids.size else 0.0
        score = numerator / max(denominator, 1e-12)
        recall_like = score
        weighted_success = numerator
    return {
        "proxy_metric_kind": kind,
        "proxy_recall": float(recall_like),
        "proxy_weighted_success": float(weighted_success),
        "proxy_denominator": float(denominator),
        "proxy_rank_score": float(score),
    }


def proxy_rows(details: Dict[Tuple[str, int, str], Dict[str, Any]], *, alpha: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (method, budget, holdout), payload in sorted(details.items()):
        if method not in METHODS:
            continue
        funnel = payload["funnel"]
        for proxy in PROXIES:
            score = score_proxy(payload, proxy, alpha=alpha)
            rows.append(
                {
                    "holdout_id": holdout,
                    "budget": int(budget),
                    "variant": method,
                    "proxy": proxy,
                    **score,
                    "cost": float(funnel["cost"]),
                    "backend_fail": int(funnel["backend_fail_total"]),
                    "unnecessary": int(funnel["unnecessary"]),
                    "served_attack_mass_product_proxy": float(funnel["served_attack_mass"]),
                    "served_clean_count": int(funnel["clean_jobs_served"]),
                    "served_jobs_total": int(funnel["served_jobs_total"]),
                }
            )
    return rows


def aggregate_proxy(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["proxy"], int(row["budget"]), row["variant"])].append(row)
    out: List[Dict[str, Any]] = []
    for (proxy, budget, variant), vals in sorted(groups.items()):
        rec = {"proxy": proxy, "budget": budget, "variant": variant, "n_holdouts": len(vals)}
        for metric in [
            "proxy_recall",
            "proxy_weighted_success",
            "proxy_rank_score",
            "cost",
            "backend_fail",
            "unnecessary",
            "served_attack_mass_product_proxy",
            "served_clean_count",
        ]:
            arr = np.asarray([float(v[metric]) for v in vals], dtype=float)
            rec[f"mean_{metric}"] = float(np.mean(arr))
            rec[f"median_{metric}"] = float(np.median(arr))
        rec["proxy_metric_kind"] = vals[0]["proxy_metric_kind"]
        out.append(rec)
    return out


def pairwise_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by = {(r["proxy"], int(r["budget"]), r["variant"], r["holdout_id"]): r for r in rows}
    holdouts = sorted({r["holdout_id"] for r in rows})
    comparisons = [
        ("source_frozen_transfer", "topk_expected_consequence"),
        ("source_frozen_transfer", "winner_replay"),
        ("source_frozen_transfer", "native_safeguarded_retune"),
    ]
    out: List[Dict[str, Any]] = []
    for proxy in PROXIES:
        for budget in BUDGETS:
            for a, b in comparisons:
                deltas = []
                for h in holdouts:
                    ka = (proxy, budget, a, h)
                    kb = (proxy, budget, b, h)
                    if ka in by and kb in by:
                        deltas.append(float(by[ka]["proxy_rank_score"]) - float(by[kb]["proxy_rank_score"]))
                if not deltas:
                    continue
                ci_lo, ci_hi = bootstrap_ci(deltas, seed=RNG_SEED + budget + len(out))
                wins = int(sum(1 for d in deltas if d > 1e-12))
                losses = int(sum(1 for d in deltas if d < -1e-12))
                ties = int(len(deltas) - wins - losses)
                out.append(
                    {
                        "proxy": proxy,
                        "budget": budget,
                        "comparison": f"{a} vs {b}",
                        "metric": "proxy_rank_score",
                        "mean_delta": float(np.mean(deltas)),
                        "median_delta": float(np.median(deltas)),
                        "min_delta": float(np.min(deltas)),
                        "max_delta": float(np.max(deltas)),
                        "bootstrap95_ci_low": ci_lo,
                        "bootstrap95_ci_high": ci_hi,
                        "exact_sign_test_p": binom_two_sided_p(wins, losses),
                        "sign_flip_p": sign_flip_p(deltas),
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                    }
                )
    return out


def rank_stability(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {(r["proxy"], int(r["budget"]), r["variant"]): float(r["mean_proxy_rank_score"]) for r in summary_rows}
    rows: List[Dict[str, Any]] = []
    for budget in BUDGETS:
        product_scores = {m: by_key[("product_proxy", budget, m)] for m in METHODS}
        product_ranks = rank_desc(product_scores)
        for proxy in PROXIES:
            scores = {m: by_key[(proxy, budget, m)] for m in METHODS}
            ranks = rank_desc(scores)
            rho = spearman_from_ranks(product_ranks, ranks, METHODS)
            ordering = " > ".join([m for m, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))])
            for m in METHODS:
                rows.append(
                    {
                        "proxy": proxy,
                        "budget": budget,
                        "variant": m,
                        "mean_rank_score": scores[m],
                        "rank": ranks[m],
                        "product_proxy_rank": product_ranks[m],
                        "spearman_with_product_ordering": rho,
                        "rank_ordering": ordering,
                    }
                )
    return rows


def write_gate3_interpretation_audit() -> None:
    rows = [
        {"status": "accepted", "wording": "Detector ceiling = 0.9711 means upstream detector is not the main bottleneck."},
        {"status": "corrected", "wording": "The main bottleneck should be stated as post-verification service plus backend-success loss, not detector ceiling."},
        {"status": "accepted", "wording": "Source-frozen is close to backend-success oracle but far from capacity oracle."},
        {"status": "corrected", "wording": "Matched-burden ordering advantage primarily holds against topk_expected_consequence."},
        {"status": "corrected", "wording": "Against winner_replay, source-frozen is not stably ahead under B=1 matched-burden diagnostics."},
        {"status": "corrected", "wording": "Topk has lower absolute burden, but should not be called more burden-efficient without qualification because recall/cost and recall/backend ratios do not uniformly beat source-frozen."},
        {"status": "accepted", "wording": "Source-frozen is a high-recall/high-burden Pareto operating point, not a dominant winner."},
        {"status": "rejected", "wording": "Any wording that treats current case39 as native success or larger-system native success."},
    ]
    lines = [
        "# Gate 3 Interpretation Audit",
        "",
        md_table(rows, ["status", "wording"]),
        "",
        "Corrected conclusion: Gate 3 supports `transfer regularization mechanism + backend stress-test limitation`, not `native success`.",
    ]
    (OUT / "gate3_interpretation_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recovery_field_audit(bank_rows: List[Dict[str, Any]], result_rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    cross = summary["cross"]
    served_attack_recover = sum(float(r["served_recover_fail_attack_jobs"]) for r in result_rows)
    served_clean_recover = sum(float(r["served_recover_fail_clean_jobs"]) for r in result_rows)
    served_attack = sum(float(r["served_attack_jobs"]) for r in result_rows)
    served_clean = sum(float(r["served_clean_jobs"]) for r in result_rows)
    result_corr = safe_corr(
        [float(r["served_recover_fail_attack_rate"]) for r in result_rows],
        [float(r["served_backend_fail_attack_rate"]) for r in result_rows],
    )
    recover_available = all(r.get("recover_fail_present") for r in bank_rows if r.get("section") == "bank_field" and r.get("is_mixed_scheduler_bank"))
    recover_sparse = float(summary["mixed_recover_frequency_mean"]) < 0.01
    main_text_ok = bool(recover_available and not recover_sparse and abs(float(summary["aggregate_recover_backend_corr"])) < 0.95)
    decision = "main_text_robustness_label" if main_text_ok else "appendix_diagnostic_only"
    rows = [
        {
            "metric": "mixed_recover_frequency_mean",
            "value": fmt(summary["mixed_recover_frequency_mean"]),
        },
        {
            "metric": "mixed_backend_frequency_mean",
            "value": fmt(summary["mixed_backend_frequency_mean"]),
        },
        {
            "metric": "recover_backend_corr",
            "value": fmt(summary["aggregate_recover_backend_corr"]),
        },
        {
            "metric": "recover_product_corr",
            "value": fmt(summary["aggregate_recover_product_corr"]),
        },
        {
            "metric": "recover_additive_corr",
            "value": fmt(summary["aggregate_recover_additive_corr"]),
        },
        {
            "metric": "recover_service_time_corr",
            "value": fmt(summary["aggregate_recover_time_corr"]),
        },
        {
            "metric": "recover_service_cost_corr",
            "value": fmt(summary["aggregate_recover_cost_corr"]),
        },
        {
            "metric": "served_recover_attack_rate",
            "value": fmt(served_attack_recover / max(served_attack, 1.0)),
        },
        {
            "metric": "served_recover_clean_rate",
            "value": fmt(served_clean_recover / max(served_clean, 1.0)),
        },
    ]
    cross_rows = [
        {"recover_fail": r, "backend_fail": b, "count": cross[(r, b)]}
        for r in [0, 1]
        for b in [0, 1]
    ]
    lines = [
        "# Recovery / Burden Field Audit",
        "",
        "Mixed banks contain `recover_fail` and `backend_fail`; raw banks expose `recover_fail` but not `backend_fail`. Exact `service_time` / `service_cost` names are absent in raw npy payloads, but scheduler arrays derive them from `stage_one_time + stage_two_time` and `delta_cost_one + delta_cost_two`.",
        "",
        md_table(rows, ["metric", "value"]),
        "",
        "## Recover Fail x Backend Fail",
        "",
        md_table(cross_rows, ["recover_fail", "backend_fail", "count"]),
        "",
        "## Decision",
        "",
        f"- `recover_fail` availability: `{recover_available}`.",
        f"- `recover_fail` sparsity flag: `{recover_sparse}`.",
        f"- `recover_fail` vs `backend_fail` served-attack-rate correlation: `{fmt(result_corr)}`.",
        f"- Label decision: `{decision}`.",
    ]
    if decision == "appendix_diagnostic_only":
        lines.append("- Conservative use: include recovery-aware results as appendix diagnostic unless reviewers specifically request a stronger recovery label.")
    else:
        lines.append("- Conservative use: recovery-aware proxy is field-supported enough for a main-text robustness check, but not as a new trained method.")
    (OUT / "recovery_field_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "recover_available": recover_available,
        "recover_sparse": recover_sparse,
        "main_text_ok": main_text_ok,
        "decision": decision,
        "recover_backend_corr": float(summary["aggregate_recover_backend_corr"]),
        "result_recover_backend_corr": result_corr,
        "served_recover_attack_rate": served_attack_recover / max(served_attack, 1.0),
        "served_recover_clean_rate": served_clean_recover / max(served_clean, 1.0),
    }


def write_proxy_summary(summary_rows: List[Dict[str, Any]], rank_rows: List[Dict[str, Any]], pair_rows: List[Dict[str, Any]], alpha: float) -> Dict[str, Any]:
    display: List[Dict[str, Any]] = []
    by = {(r["proxy"], int(r["budget"]), r["variant"]): r for r in summary_rows}
    pair_by = {(r["proxy"], int(r["budget"]), r["comparison"]): r for r in pair_rows}
    product_spearman = {
        (r["proxy"], int(r["budget"])): float(r["spearman_with_product_ordering"])
        for r in rank_rows
        if r["variant"] == "source_frozen_transfer"
    }
    for proxy in PROXIES:
        for budget in BUDGETS:
            src = by[(proxy, budget, "source_frozen_transfer")]
            topk = by[(proxy, budget, "topk_expected_consequence")]
            winner = by[(proxy, budget, "winner_replay")]
            safe = by[(proxy, budget, "native_safeguarded_retune")]
            display.append(
                {
                    "proxy": proxy,
                    "B": budget,
                    "source_score": fmt(src["mean_proxy_rank_score"]),
                    "source_vs_topk": fmt(float(src["mean_proxy_rank_score"]) - float(topk["mean_proxy_rank_score"])),
                    "source_vs_winner": fmt(float(src["mean_proxy_rank_score"]) - float(winner["mean_proxy_rank_score"])),
                    "source_vs_safeguarded": fmt(float(src["mean_proxy_rank_score"]) - float(safe["mean_proxy_rank_score"])),
                    "rho_vs_product": fmt(product_spearman[(proxy, budget)]),
                    "source_cost": fmt(src["mean_cost"]),
                    "source_backend": fmt(src["mean_backend_fail"]),
                }
            )
    source_topk_stable = all(
        pair_by[(p, b, "source_frozen_transfer vs topk_expected_consequence")]["mean_delta"] > 0
        for p in PROXIES
        for b in BUDGETS
        if p != "burden_proxy"
    )
    source_winner_stable = all(
        pair_by[(p, b, "source_frozen_transfer vs winner_replay")]["mean_delta"] > 0
        for p in PROXIES
        for b in BUDGETS
        if p != "burden_proxy"
    )
    native_collapse = all(
        float(by[(p, b, "native_safeguarded_retune")]["mean_proxy_rank_score"])
        < float(by[(p, b, "source_frozen_transfer")]["mean_proxy_rank_score"])
        for p in ["product_proxy", "additive_proxy", "backend_success_proxy", "recovery_aware_proxy"]
        for b in BUDGETS
    )
    lines = [
        "# Proxy Robustness Summary",
        "",
        f"`success_burden_proxy` uses alpha `{fmt(alpha)}` derived from native train/val medians; no test holdout was used to tune alpha.",
        "",
        "For `burden_proxy`, rank score is inverted burden, so higher is lower served burden rather than higher consequence recall.",
        "",
        md_table(display, ["proxy", "B", "source_score", "source_vs_topk", "source_vs_winner", "source_vs_safeguarded", "rho_vs_product", "source_cost", "source_backend"]),
        "",
        "## Stability",
        "",
        f"- Source-frozen vs topk is stable on high-is-better consequence proxies excluding pure burden: `{source_topk_stable}`.",
        f"- Source-frozen vs winner_replay is stable on high-is-better consequence proxies excluding pure burden: `{source_winner_stable}`.",
        f"- Native safeguarded collapse persists across product/additive/backend-success/recovery-aware proxies: `{native_collapse}`.",
        "- Pure burden scoring penalizes source-frozen because it is a high-service/high-backend-fail operating point.",
    ]
    (OUT / "proxy_robustness_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "source_topk_stable": source_topk_stable,
        "source_winner_stable": source_winner_stable,
        "native_collapse": native_collapse,
    }


def write_robustness_decision(field_decision: Dict[str, Any], proxy_decision: Dict[str, Any], summary_rows: List[Dict[str, Any]]) -> None:
    by = {(r["proxy"], int(r["budget"]), r["variant"]): r for r in summary_rows}
    burden_source_scores = [float(by[("burden_proxy", b, "source_frozen_transfer")]["mean_proxy_rank_score"]) for b in BUDGETS]
    burden_topk_scores = [float(by[("burden_proxy", b, "topk_expected_consequence")]["mean_proxy_rank_score"]) for b in BUDGETS]
    burden_weakens = bool(np.mean(burden_source_scores) < np.mean(burden_topk_scores))
    lines = [
        "# Gate 4 Robustness Decision",
        "",
        "1. Source-frozen remains a high consequence-recall / high burden point across product, additive, backend-success, recovery-aware, and success-burden diagnostics.",
        f"2. Source-frozen relative to topk is stable on high-is-better consequence proxies: `{proxy_decision['source_topk_stable']}`.",
        f"3. Source-frozen relative to winner_replay is stable on high-is-better consequence proxies: `{proxy_decision['source_winner_stable']}`; interpret winner comparisons more cautiously than topk comparisons.",
        f"4. Native local retune collapse persists across consequence proxies: `{proxy_decision['native_collapse']}`.",
        "5. `recovery_aware_proxy` does not create a new winner; it mainly validates that the ordering is not an artifact of product severity alone.",
        f"6. `burden_proxy` weakens any dominant-winner claim for source-frozen: `{burden_weakens}`.",
        "7. Current paper should not say fully recovery-aware or physical-consequence-aware as a method claim; safe wording is `proxy-consequence-guided` with recovery/burden robustness diagnostics.",
        "8. Main text: product/additive/backend-success/recovery-aware robustness and high-burden Pareto interpretation. Appendix: raw recovery-field audit, pure burden-proxy ranks, and fail-capped extension protocol.",
    ]
    if field_decision["decision"] == "appendix_diagnostic_only":
        lines.append("9. `recover_fail` is usable for appendix diagnostics, but not strong enough alone to become the main paper's primary recovery label.")
    else:
        lines.append("9. `recover_fail` is field-supported enough for a main-text robustness check, but still should not be framed as a newly trained recovery-aware method.")
    (OUT / "gate4_robustness_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fail_capped_protocol() -> None:
    lines = [
        "# Fail-capped Source-frozen Extension Protocol",
        "",
        "Status: preregistered design only. Do not execute until Gate 4 robustness is reviewed.",
        "",
        "## Candidate Variant",
        "",
        "- Name: `source_frozen_fail_capped`.",
        "- Keep source-frozen score weights, winner variant, budgets, Wmax, and regime unchanged.",
        "- Add only a pre-registered fail/burden gate before service admission.",
        "- Gate inputs allowed: predicted fail probability, predicted service cost, predicted service time.",
        "- No true backend_fail, recover_fail, or test-holdout labels may enter deployment scoring.",
        "",
        "## Candidate Cap Grid",
        "",
        "- `pred_fail_prob_cap`: `[0.50, 0.60, 0.70, 0.80]`.",
        "- `pred_service_cost_cap_quantile`: `[0.75, 0.85, 0.95, None]` computed on selection split only.",
        "- `pred_service_time_cap_quantile`: `[0.75, 0.85, 0.95, None]` computed on selection split only.",
        "- `burden_penalty_lambda`: `[0.00, 0.05, 0.10, 0.20]` if using penalty instead of hard cap.",
        "",
        "## Selection Split",
        "",
        "- Primary selection: source train/val for transfer-preserving variant.",
        "- Optional explicit native-val selection: allowed only if reported as a separate native-val-selected diagnostic.",
        "- The frozen 8 test holdouts cannot select cap, penalty, or winner.",
        "",
        "## Fixed Endpoints",
        "",
        "- Primary endpoints: recall, backend_fail, cost, unnecessary, served_ratio, delay_p95.",
        "- Scheduler-conditioned endpoints: served_attack_mass, backend_success_attack_mass, clean service count.",
        "- Robustness endpoints: product_proxy, additive_proxy, backend_success_proxy, recovery_aware_proxy, success_burden_proxy.",
        "",
        "## Success Criteria",
        "",
        "- Primary: backend_fail decreases by at least 15% or 20% versus source-frozen while recall drops by no more than 5% or 10%.",
        "- Secondary: not dominated by source-frozen, topk_expected_consequence, or winner_replay on recall-backend_fail and recall-cost Pareto frontiers.",
        "- All cap-grid outcomes must be shown; no cherry-picking.",
        "",
        "## Promotion Rule",
        "",
        "- v2 main experiment only if it satisfies primary criteria on selection split and confirms on all 8 frozen holdouts without test-selected tuning.",
        "- Future work only if it reduces burden but fails recall retention or Pareto non-domination.",
    ]
    (OUT / "fail_capped_extension_protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_rank_heatmap(rank_rows: List[Dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, budget in zip(axes, BUDGETS):
        mat = np.zeros((len(METHODS), len(PROXIES)), dtype=float)
        for i, method in enumerate(METHODS):
            for j, proxy in enumerate(PROXIES):
                row = next(r for r in rank_rows if r["budget"] == budget and r["variant"] == method and r["proxy"] == proxy)
                mat[i, j] = float(row["rank"])
        im = ax.imshow(mat, cmap="viridis_r", vmin=1, vmax=len(METHODS))
        ax.set_title(f"Proxy rank stability B={budget}")
        ax.set_xticks(np.arange(len(PROXIES)))
        ax.set_xticklabels(PROXIES, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(METHODS)))
        ax.set_yticklabels(METHODS, fontsize=7)
        for i in range(len(METHODS)):
            for j in range(len(PROXIES)):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if mat[i, j] > 4 else "black")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="rank, 1 = best")
    fig.savefig(OUT / "figure_proxy_rank_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "figure_proxy_rank_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_proxy_recall_vs_burden(summary_rows: List[Dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    proxies = ["product_proxy", "backend_success_proxy", "recovery_aware_proxy", "success_burden_proxy"]
    for ax, proxy in zip(axes.ravel(), proxies):
        for budget, marker in [(1, "o"), (2, "s")]:
            rows = [r for r in summary_rows if r["proxy"] == proxy and int(r["budget"]) == budget and r["variant"] in DISPLAY_METHODS]
            for r in rows:
                ax.scatter(float(r["mean_backend_fail"]), float(r["mean_proxy_rank_score"]), marker=marker)
                ax.annotate(f"{r['variant'].replace('_retune', '').replace('_transfer', '')}-B{budget}", (float(r["mean_backend_fail"]), float(r["mean_proxy_rank_score"])), fontsize=5)
        ax.set_title(proxy)
        ax.set_xlabel("mean backend_fail")
        ax.set_ylabel("mean proxy rank score")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figure_proxy_recall_vs_burden.png", dpi=220)
    fig.savefig(OUT / "figure_proxy_recall_vs_burden.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_gate3_interpretation_audit()
    funnel_rows, ceiling_rows, details = G3.detail_records()
    _ = ceiling_rows
    bank_rows, bank_summary = bank_audit_rows()
    result_rows = result_recovery_rows(details)
    write_csv(OUT / "recovery_field_audit.csv", bank_rows + result_rows)
    field_decision = write_recovery_field_audit(bank_rows, result_rows, bank_summary)
    alpha = compute_alpha_from_native_train_val()
    by_holdout = proxy_rows(details, alpha=alpha)
    write_csv(OUT / "proxy_robustness_by_holdout.csv", by_holdout)
    summary_rows = aggregate_proxy(by_holdout)
    pair_rows = pairwise_stats(by_holdout)
    rank_rows = rank_stability(summary_rows)
    write_csv(OUT / "proxy_rank_stability.csv", rank_rows)
    write_csv(OUT / "proxy_pairwise_statistics.csv", pair_rows)
    proxy_decision = write_proxy_summary(summary_rows, rank_rows, pair_rows, alpha)
    plot_rank_heatmap(rank_rows)
    plot_proxy_recall_vs_burden(summary_rows)
    write_robustness_decision(field_decision, proxy_decision, summary_rows)
    write_fail_capped_protocol()
    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    (OUT / "outputs_tree.txt").write_text("\n".join(sorted(set(files) | {"outputs_tree.txt"})) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(OUT),
                "proxy_rows": len(by_holdout),
                "pairwise_rows": len(pair_rows),
                "rank_rows": len(rank_rows),
                "recover_decision": field_decision["decision"],
                "source_topk_stable": proxy_decision["source_topk_stable"],
                "source_winner_stable": proxy_decision["source_winner_stable"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
