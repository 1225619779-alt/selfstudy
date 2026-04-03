
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


EPS = 1e-12


@dataclass
class BinnedStatisticModel:
    edges: np.ndarray
    values: np.ndarray
    default_value: float
    name: str = "binned_model"

    def predict(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        out = np.full(arr.shape, self.default_value, dtype=float)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return out
        idx = np.digitize(arr[finite], self.edges[1:-1], right=False)
        idx = np.clip(idx, 0, len(self.values) - 1)
        out[finite] = self.values[idx]
        return out

    def predict_scalar(self, x: float) -> float:
        return float(self.predict(np.asarray([x], dtype=float))[0])

    def to_dict(self) -> Dict[str, object]:
        return {
            "edges": self.edges,
            "values": self.values,
            "default_value": float(self.default_value),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BinnedStatisticModel":
        return cls(
            edges=np.asarray(payload["edges"], dtype=float),
            values=np.asarray(payload["values"], dtype=float),
            default_value=float(payload["default_value"]),
            name=str(payload.get("name", "binned_model")),
        )


def _finite_pair(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def _make_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    if x.size == 0:
        return np.asarray([-np.inf, np.inf], dtype=float)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.asarray([-np.inf, np.inf], dtype=float)
    q = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    inner = np.quantile(x, q)
    inner = np.unique(inner)
    if inner.size < 2:
        spread = max(abs(float(inner[0])) * 0.1, 1e-6)
        inner = np.asarray([float(inner[0]) - spread, float(inner[0]) + spread], dtype=float)
    edges = inner.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def fit_binned_mean(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 20,
    min_count: int = 3,
    default_value: float | None = None,
    name: str = "binned_mean",
) -> BinnedStatisticModel:
    x_arr, y_arr = _finite_pair(x, y)
    if default_value is None:
        default_value = float(np.nanmean(y_arr)) if y_arr.size else 0.0
    edges = _make_edges(x_arr, n_bins=n_bins)
    n_real_bins = len(edges) - 1
    values = np.full(n_real_bins, float(default_value), dtype=float)
    if x_arr.size == 0:
        return BinnedStatisticModel(edges=edges, values=values, default_value=float(default_value), name=name)
    idx = np.digitize(x_arr, edges[1:-1], right=False)
    idx = np.clip(idx, 0, n_real_bins - 1)
    for b in range(n_real_bins):
        mask = idx == b
        if int(mask.sum()) >= int(min_count):
            values[b] = float(np.mean(y_arr[mask]))
    return BinnedStatisticModel(edges=edges, values=values, default_value=float(default_value), name=name)


def fit_binned_posterior(
    x_neg: Sequence[float] | np.ndarray,
    x_pos: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 20,
    laplace_alpha: float = 1.0,
    min_count: int = 1,
    name: str = "binned_posterior",
) -> BinnedStatisticModel:
    x0 = np.asarray(x_neg, dtype=float).reshape(-1)
    x1 = np.asarray(x_pos, dtype=float).reshape(-1)
    x0 = x0[np.isfinite(x0)]
    x1 = x1[np.isfinite(x1)]
    x_all = np.concatenate([x0, x1]) if (x0.size + x1.size) > 0 else np.asarray([], dtype=float)
    edges = _make_edges(x_all, n_bins=n_bins)
    n_real_bins = len(edges) - 1
    values = np.full(n_real_bins, 0.5, dtype=float)
    if x_all.size == 0:
        return BinnedStatisticModel(edges=edges, values=values, default_value=0.5, name=name)
    idx0 = np.digitize(x0, edges[1:-1], right=False)
    idx1 = np.digitize(x1, edges[1:-1], right=False)
    idx0 = np.clip(idx0, 0, n_real_bins - 1)
    idx1 = np.clip(idx1, 0, n_real_bins - 1)
    global_default = float((x1.size + laplace_alpha) / (x0.size + x1.size + 2.0 * laplace_alpha))
    for b in range(n_real_bins):
        n0 = int(np.sum(idx0 == b))
        n1 = int(np.sum(idx1 == b))
        total = n0 + n1
        if total >= int(min_count):
            values[b] = float((n1 + laplace_alpha) / (total + 2.0 * laplace_alpha))
        else:
            values[b] = global_default
    return BinnedStatisticModel(edges=edges, values=values, default_value=global_default, name=name)


def _load_metric(path: str) -> Dict[str, object]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict-like metric at {path}, got {type(payload)!r}")
    return payload


def _flatten_group_dict(group_dict: Mapping[str, Sequence[float] | np.ndarray]) -> np.ndarray:
    values: List[np.ndarray] = []
    for _, arr in group_dict.items():
        arr_np = np.asarray(arr, dtype=float).reshape(-1)
        values.append(arr_np)
    if not values:
        return np.asarray([], dtype=float)
    return np.concatenate(values, axis=0)


def load_alarm_bank_signal(metric_path: str, signal_key: str) -> np.ndarray:
    metric = _load_metric(metric_path)
    if signal_key not in metric:
        raise KeyError(f"{signal_key!r} not found in {metric_path}")
    group_dict = metric[signal_key]
    if not isinstance(group_dict, Mapping):
        raise TypeError(f"Expected mapping for {signal_key!r} in {metric_path}")
    return _flatten_group_dict(group_dict)


def fit_attack_posterior_from_banks(
    clean_metric_path: str,
    attack_metric_path: str,
    *,
    signal_key: str = "score_phys_l2",
    n_bins: int = 20,
) -> BinnedStatisticModel:
    clean_signal = load_alarm_bank_signal(clean_metric_path, signal_key)
    attack_signal = load_alarm_bank_signal(attack_metric_path, signal_key)
    return fit_binned_posterior(
        x_neg=clean_signal,
        x_pos=attack_signal,
        n_bins=n_bins,
        name=f"posterior_{signal_key}",
    )


def mixed_bank_to_alarm_arrays(mixed_metric_path: str) -> Dict[str, np.ndarray]:
    metric = _load_metric(mixed_metric_path)
    required = [
        "timeline_step",
        "ddd_alarm",
        "verify_score",
        "ddd_loss_recons",
        "is_attack_step",
        "ang_no_summary",
        "ang_str_summary",
        "stage_one_time",
        "stage_two_time",
        "delta_cost_one",
        "delta_cost_two",
        "recover_fail",
        "backend_fail",
    ]
    missing = [k for k in required if k not in metric]
    if missing:
        raise KeyError(f"Missing keys in mixed metric {mixed_metric_path}: {missing}")

    arr = {k: np.asarray(metric[k]) for k in required}
    ddd_mask = np.asarray(arr["ddd_alarm"], dtype=int).reshape(-1) == 1

    out: Dict[str, np.ndarray] = {}
    out["arrival_step"] = np.asarray(arr["timeline_step"], dtype=int).reshape(-1)[ddd_mask]
    out["verify_score"] = np.asarray(arr["verify_score"], dtype=float).reshape(-1)[ddd_mask]
    out["ddd_loss_recons"] = np.asarray(arr["ddd_loss_recons"], dtype=float).reshape(-1)[ddd_mask]
    out["is_attack"] = np.asarray(arr["is_attack_step"], dtype=int).reshape(-1)[ddd_mask]
    out["ang_no"] = np.asarray(arr["ang_no_summary"], dtype=float).reshape(-1)[ddd_mask]
    out["ang_str"] = np.asarray(arr["ang_str_summary"], dtype=float).reshape(-1)[ddd_mask]
    out["recover_fail"] = np.asarray(arr["recover_fail"], dtype=int).reshape(-1)[ddd_mask]
    out["backend_fail"] = np.asarray(arr["backend_fail"], dtype=int).reshape(-1)[ddd_mask]
    t1 = np.asarray(arr["stage_one_time"], dtype=float).reshape(-1)[ddd_mask]
    t2 = np.asarray(arr["stage_two_time"], dtype=float).reshape(-1)[ddd_mask]
    c1 = np.asarray(arr["delta_cost_one"], dtype=float).reshape(-1)[ddd_mask]
    c2 = np.asarray(arr["delta_cost_two"], dtype=float).reshape(-1)[ddd_mask]
    out["service_time"] = np.nan_to_num(t1, nan=0.0, posinf=0.0, neginf=0.0) + np.nan_to_num(t2, nan=0.0, posinf=0.0, neginf=0.0)
    out["service_cost"] = np.nan_to_num(c1, nan=0.0, posinf=0.0, neginf=0.0) + np.nan_to_num(c2, nan=0.0, posinf=0.0, neginf=0.0)

    # Keep the truth severity intentionally separate from observable signals.
    # For attack jobs we only use ang_no/ang_str to define evaluation-side consequence.
    severity_true = np.maximum(out["ang_no"], 0.0) * np.maximum(out["ang_str"], 0.0)
    out["severity_true"] = np.where(out["is_attack"] == 1, severity_true, 0.0)

    # Optional richer consequence proxy if the mixed bank already stores one.
    optional_keys = [
        "consequence_proxy",
        "value_proxy",
        "score_phys_l2",
        "recover_time_alarm",
    ]
    for key in optional_keys:
        if key in metric:
            out[key] = np.asarray(metric[key], dtype=float).reshape(-1)[ddd_mask]

    summary = metric.get("summary", {})
    total_steps = int(summary.get("total_steps", int(np.max(out["arrival_step"]) + 1 if out["arrival_step"].size else 0)))
    out["total_steps"] = np.asarray([total_steps], dtype=int)
    return out


def fit_service_models_from_mixed_bank(
    mixed_metric_path: str,
    *,
    signal_key: str = "verify_score",
    n_bins: int = 20,
) -> Dict[str, BinnedStatisticModel]:
    arr = mixed_bank_to_alarm_arrays(mixed_metric_path)
    if signal_key not in arr:
        raise KeyError(f"signal_key={signal_key!r} not present in mixed bank arrays")
    x = np.asarray(arr[signal_key], dtype=float)
    return {
        "service_time": fit_binned_mean(x, arr["service_time"], n_bins=n_bins, name=f"time_given_{signal_key}"),
        "service_cost": fit_binned_mean(x, arr["service_cost"], n_bins=n_bins, name=f"cost_given_{signal_key}"),
        "backend_fail": fit_binned_mean(x, arr["backend_fail"], n_bins=n_bins, name=f"fail_given_{signal_key}"),
    }


def fit_attack_severity_models_from_arrays(
    arr: Dict[str, np.ndarray],
    *,
    signal_keys: Sequence[str] = ("verify_score", "ddd_loss_recons"),
    n_bins: int = 20,
) -> Dict[str, BinnedStatisticModel]:
    """Fit E[severity_true | signal, attack==1] on attack jobs only."""
    models: Dict[str, BinnedStatisticModel] = {}
    attack_mask = np.asarray(arr["is_attack"], dtype=int) == 1
    y = np.asarray(arr["severity_true"], dtype=float)[attack_mask]
    for key in signal_keys:
        if key not in arr:
            continue
        x = np.asarray(arr[key], dtype=float)[attack_mask]
        default_value = float(np.mean(y)) if y.size else 0.0
        models[key] = fit_binned_mean(
            x,
            y,
            n_bins=n_bins,
            default_value=default_value,
            name=f"attack_severity_given_{key}",
        )
    return models


def fit_expected_consequence_models_from_arrays(
    arr: Dict[str, np.ndarray],
    *,
    signal_keys: Sequence[str] = ("verify_score", "ddd_loss_recons"),
    n_bins: int = 20,
) -> Dict[str, BinnedStatisticModel]:
    """Fit E[severity_true | signal] directly, including zeros for clean jobs."""
    models: Dict[str, BinnedStatisticModel] = {}
    y = np.asarray(arr["severity_true"], dtype=float)
    for key in signal_keys:
        if key not in arr:
            continue
        x = np.asarray(arr[key], dtype=float)
        models[key] = fit_binned_mean(x, y, n_bins=n_bins, name=f"expected_consequence_given_{key}")
    return models


def summarize_array(x: Sequence[float] | np.ndarray) -> Dict[str, float]:
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0.0}
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }
