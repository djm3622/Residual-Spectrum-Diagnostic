"""Shared metric aggregation helpers used by run scripts."""

from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np


def safe_mean(values: List[float]) -> float:
    """Mean over finite values, preserving NaN only when all values are non-finite."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def safe_pearson_corr(x: List[float], y: List[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")

    x_valid = x_arr[mask]
    y_valid = y_arr[mask]
    x_center = x_valid - np.mean(x_valid)
    y_center = y_valid - np.mean(y_valid)
    denom = float(np.sqrt(np.sum(x_center * x_center) * np.sum(y_center * y_center)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x_center * y_center) / denom)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    order = np.argsort(arr, kind="mergesort")
    sorted_arr = arr[order]
    ranks = np.empty(arr.size, dtype=np.float64)

    start = 0
    while start < arr.size:
        end = start + 1
        while end < arr.size and sorted_arr[end] == sorted_arr[start]:
            end += 1
        avg_rank = 0.5 * float(start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def safe_spearman_corr(x: List[float], y: List[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")

    x_rank = rankdata_average(x_arr[mask])
    y_rank = rankdata_average(y_arr[mask])
    return safe_pearson_corr(x_rank.tolist(), y_rank.tolist())


def build_metric_vs_l2(
    metrics: Mapping[str, List[float]],
    metric_ids: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    output: Dict[str, Dict[str, Dict[str, float]]] = {"clean": {}, "noisy": {}}
    for split in ("clean", "noisy"):
        l2_key = f"{split}_l2"
        l2_values = list(metrics.get(l2_key, []))
        for metric_id in metric_ids:
            metric_key = f"{split}_{metric_id}"
            metric_values = list(metrics.get(metric_key, []))
            x_arr = np.asarray(l2_values, dtype=np.float64)
            y_arr = np.asarray(metric_values, dtype=np.float64)
            valid_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            output[split][metric_id] = {
                "pearson": safe_pearson_corr(l2_values, metric_values),
                "spearman": safe_spearman_corr(l2_values, metric_values),
                "n": int(np.count_nonzero(valid_mask)),
            }
    return output


def build_paired_bootstrap_gap(
    clean_values: List[float],
    noisy_values: List[float],
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    rng_seed: int = 0,
) -> Dict[str, float]:
    """Bootstrap CI for paired gap: noisy - clean."""
    clean_arr = np.asarray(clean_values, dtype=np.float64).reshape(-1)
    noisy_arr = np.asarray(noisy_values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(clean_arr) & np.isfinite(noisy_arr)
    clean = clean_arr[valid]
    noisy = noisy_arr[valid]
    gap = noisy - clean

    n = int(gap.size)
    result: Dict[str, float] = {
        "clean_mean": float(np.mean(clean)) if n > 0 else float("nan"),
        "noisy_mean": float(np.mean(noisy)) if n > 0 else float("nan"),
        "gap_mean": float(np.mean(gap)) if n > 0 else float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "boot_std": float("nan"),
        "prob_gap_gt_zero": float("nan"),
        "prob_gap_lt_zero": float("nan"),
        "n": float(n),
        "n_bootstrap": float(max(1, int(n_bootstrap))),
        "ci_level": float(ci_level),
    }
    if n <= 0:
        return result

    draws = max(1, int(n_bootstrap))
    if n == 1:
        boot_means = np.full((draws,), float(gap[0]), dtype=np.float64)
    else:
        rng = np.random.default_rng(int(rng_seed))
        sample_idx = rng.integers(0, n, size=(draws, n))
        boot_means = np.mean(gap[sample_idx], axis=1)

    alpha = 0.5 * max(0.0, min(1.0, 1.0 - float(ci_level)))
    low_q = alpha
    high_q = 1.0 - alpha
    result["ci_low"] = float(np.quantile(boot_means, low_q))
    result["ci_high"] = float(np.quantile(boot_means, high_q))
    result["boot_std"] = float(np.std(boot_means, ddof=0))
    result["prob_gap_gt_zero"] = float(np.mean(boot_means > 0.0))
    result["prob_gap_lt_zero"] = float(np.mean(boot_means < 0.0))
    return result
