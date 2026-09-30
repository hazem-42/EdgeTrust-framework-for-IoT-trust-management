"""
metrics.py -- every metric named in the Performance Evaluation section:
confusion matrix and its derived metrics, ROC-AUC, per-category recall,
state-flap rate, the corrected Theorem 5 resilience-bound check, and
latency/throughput measurement helpers.
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
)


def detection_metrics(y_true, y_pred, y_score=None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "confusion_matrix": cm.tolist(),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else float("nan"),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_score)
            fpr, tpr, thr = roc_curve(y_true, y_score)
            out["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}
        except ValueError as e:
            out["roc_auc"] = None
            out["roc_curve_error"] = str(e)
    return out


def per_category_recall(y_true_category: pd.Series, y_pred_binary: np.ndarray,
                         categories: list) -> pd.Series:
    """
    For each real attack category (excluding Benign), what fraction of its
    flows were correctly flagged malicious (y_pred_binary == 1)? This is
    the metric that a strong binary accuracy can hide: a rare category can
    be almost entirely missed while overall accuracy stays high.
    """
    recalls = {}
    for cat in categories:
        if cat == "Benign":
            continue
        mask = (y_true_category == cat).to_numpy()
        n = mask.sum()
        if n == 0:
            recalls[cat] = np.nan
            continue
        recalls[cat] = float(y_pred_binary[mask].mean())
    return pd.Series(recalls, name="recall")


def state_flap_rate(states: np.ndarray) -> int:
    """Counts transitions in a state sequence (used for the hysteresis stress test)."""
    return int(np.sum(np.diff(states) != 0))


def theorem5_bound_check(weights: np.ndarray, rho: float, alpha: float,
                          eta: float, eps_values: np.ndarray,
                          n_trials: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    Empirically verifies the CORRECTED Theorem 5 bound:
        Delta_T <= eps * m * w_max * eta / (1 + rho + alpha)
    against the empirical worst-case trust-score deviation under a
    rank-based worst-case feature perturbation. Uses this run's OWN
    fitted entropy weights, not a synthetic reference window.
    """
    rng = np.random.default_rng(seed)
    m = len(weights)
    w_max = weights.max()
    rows = []
    for eps in eps_values:
        k = max(1, int(round(eps * m)))
        idx = np.argsort(-weights)[:k]
        deltas = []
        for _ in range(n_trials):
            r_base = rng.uniform(0.3, 0.9, size=m)
            T_base = (np.dot(weights, r_base) + rho * 0.5 + alpha * 0.95) / (1 + rho + alpha)
            r_pert = r_base.copy()
            r_pert[idx] = np.clip(r_pert[idx] - eta, 0, 1)
            T_pert = (np.dot(weights, r_pert) + rho * 0.5 + alpha * 0.95) / (1 + rho + alpha)
            deltas.append(T_base - T_pert)
        empirical_max = max(deltas)
        bound = k * w_max * eta / (1 + rho + alpha)
        rows.append({"eps": eps, "empirical_max_delta": empirical_max,
                      "corrected_bound": bound, "bound_holds": empirical_max <= bound + 1e-9})
    return pd.DataFrame(rows)


def measure_latency(fn, *args, n_repeats: int = 1000, **kwargs) -> dict:
    """
    Wall-clock latency of a single call to fn(*args, **kwargs), repeated
    n_repeats times. Returns mean/p50/p95/p99 in milliseconds. Run this ON
    THE TARGET HARDWARE (e.g., the actual Raspberry Pi 4B), not on a dev
    laptop, if the number is meant to support a hardware latency claim.
    """
    times_ms = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times_ms)
    return {
        "mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)), "p99_ms": float(np.percentile(arr, 99)),
        "n_repeats": n_repeats,
        "raw_times_ms": arr.tolist(),
    }


def throughput_projection(measured_latency_ms: float, device_counts: np.ndarray,
                           epoch_interval_s: float = 5.0, n_threads: int = 1) -> pd.DataFrame:
    """Analytical projection of decisions/second vs. monitored-device count
    from a MEASURED per-decision latency (see measure_latency)."""
    per_decision_s = measured_latency_ms / 1000.0
    ceiling = n_threads / per_decision_s
    required = device_counts / epoch_interval_s
    achievable = np.minimum(required, ceiling)
    return pd.DataFrame({
        "devices": device_counts, "required_decisions_per_s": required,
        "ceiling_decisions_per_s": ceiling, "achievable_decisions_per_s": achievable,
    })
