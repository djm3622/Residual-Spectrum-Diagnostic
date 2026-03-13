"""Lightweight plotting helpers for single-seed runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Mapping

# Use writable cache location in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


def _shared_field_norm(
    frames: List[np.ndarray],
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> mcolors.Normalize:
    """Build a shared robust normalization for signed/unsigned field panels."""
    finite_values: List[np.ndarray] = []
    for frame in frames:
        arr = np.asarray(frame, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            finite_values.append(finite.reshape(-1))

    if not finite_values:
        return mcolors.Normalize(vmin=-1.0, vmax=1.0)

    merged = np.concatenate(finite_values)
    vmin = float(np.percentile(merged, low_pct))
    vmax = float(np.percentile(merged, high_pct))
    if not np.isfinite(vmin):
        vmin = float(np.min(merged))
    if not np.isfinite(vmax):
        vmax = float(np.max(merged))
    if vmax <= vmin:
        vmin = float(np.min(merged))
        vmax = float(np.max(merged))
    if vmax <= vmin:
        vmax = vmin + 1e-12

    if vmin < 0.0 < vmax:
        abs_lim = max(abs(vmin), abs(vmax), 1e-12)
        return mcolors.TwoSlopeNorm(vmin=-abs_lim, vcenter=0.0, vmax=abs_lim)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _annotate_min_max(ax: plt.Axes, data: np.ndarray) -> None:
    """Overlay per-panel min/max values."""
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    ax.text(
        0.02,
        0.98,
        f"min {min_val:.2e}\nmax {max_val:.2e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 2.5, "edgecolor": "none"},
    )


def _ordered_unique(labels: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _row_label_colors(labels: List[str]) -> Dict[str, tuple]:
    unique_labels = _ordered_unique(labels)
    palette = plt.get_cmap("tab20")
    return {
        label: palette(idx % palette.N)
        for idx, label in enumerate(unique_labels)
    }


def _legend_patch(cmap_name: str, label: str, value: float = 0.75) -> Patch:
    color = plt.get_cmap(cmap_name)(value)
    return Patch(facecolor=color, edgecolor="none", label=label)


def save_clean_noisy_summary_plot(
    metrics: Dict[str, float],
    title: str,
    output_path: str | Path,
) -> None:
    """Save a compact clean-vs-noisy comparison chart for one run."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["L2", "HFV", "LFV"]
    clean_values = [metrics["clean_l2"], metrics["clean_hfv"], metrics["clean_lfv"]]
    noisy_values = [metrics["noisy_l2"], metrics["noisy_hfv"], metrics["noisy_lfv"]]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar([i - width / 2 for i in x], clean_values, width=width, label="Clean", color="#0072B2")
    ax.bar([i + width / 2 for i in x], noisy_values, width=width, label="Noisy", color="#D55E00")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric value")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    fig.subplots_adjust(right=0.84)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_scalar_fit_panel(
    input_field: np.ndarray,
    target_field: np.ndarray,
    pred_field: np.ndarray,
    output_path: str | Path,
    title: str,
    cmap: str = "RdBu_r",
    input_cmap: str = "cividis",
    input_border_color: str = "#2A9D8F",
    input_border_width: float = 2.0,
    target_field_noisy: np.ndarray | None = None,
    model_label: str = "Model",
) -> None:
    """Save scalar fit diagnostics with clean/noisy ground-truth comparison."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clean_target = np.asarray(target_field, dtype=np.float32)
    noisy_target = np.asarray(target_field_noisy if target_field_noisy is not None else target_field, dtype=np.float32)
    pred = np.asarray(pred_field, dtype=np.float32)
    inp = np.asarray(input_field, dtype=np.float32)

    err_vs_clean = np.abs(pred - clean_target)
    err_vs_noisy = np.abs(pred - noisy_target)
    field_norm = _shared_field_norm([inp, clean_target, noisy_target, pred], low_pct=1.0, high_pct=99.0)

    fig, axes = plt.subplots(1, 6, figsize=(16.8, 3.2))
    panels = [
        (inp, "Input", input_cmap, "field"),
        (clean_target, "Clean Ground Truth", cmap, "field"),
        (noisy_target, "Noisy Ground Truth", cmap, "field"),
        (pred, "Predicted", cmap, "field"),
        (err_vs_clean, "Abs Error vs Clean GT", "magma", "error"),
        (err_vs_noisy, "Abs Error vs Noisy GT", "magma", "error"),
    ]

    for idx, (data, name, panel_cmap, panel_kind) in enumerate(panels):
        ax = axes[idx]
        if idx == 0:
            im = ax.imshow(data, cmap=panel_cmap, norm=field_norm)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(input_border_width)
                spine.set_edgecolor(input_border_color)
        elif panel_kind == "field":
            im = ax.imshow(data, cmap=panel_cmap, norm=field_norm)
        else:
            im = ax.imshow(data, cmap=panel_cmap)
        ax.set_title(name)
        _annotate_min_max(ax, data)
        if idx == 0:
            ax.set_ylabel(model_label, rotation=90, labelpad=8, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    legend_handles = [
        _legend_patch(input_cmap, "Input colormap"),
        _legend_patch(cmap, "Field colormap"),
        _legend_patch("magma", "Error colormap"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=True,
        fontsize=8,
    )

    fig.subplots_adjust(left=0.05, right=0.84, top=0.92, bottom=0.08, wspace=0.28)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_coupled_fit_panel(
    input_u: np.ndarray,
    input_v: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    pred_u: np.ndarray,
    pred_v: np.ndarray,
    output_path: str | Path,
    title: str,
    output_cmap: str = "viridis",
    input_cmap: str = "cividis",
    input_border_color: str = "#2A9D8F",
    input_border_width: float = 2.0,
    target_u_noisy: np.ndarray | None = None,
    target_v_noisy: np.ndarray | None = None,
    model_label: str = "Model",
) -> None:
    """Save coupled fit diagnostics with clean/noisy ground-truth comparison."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    target_u_clean = np.asarray(target_u, dtype=np.float32)
    target_v_clean = np.asarray(target_v, dtype=np.float32)
    target_u_noisy_arr = np.asarray(target_u_noisy if target_u_noisy is not None else target_u, dtype=np.float32)
    target_v_noisy_arr = np.asarray(target_v_noisy if target_v_noisy is not None else target_v, dtype=np.float32)
    pred_u_arr = np.asarray(pred_u, dtype=np.float32)
    pred_v_arr = np.asarray(pred_v, dtype=np.float32)
    input_u_arr = np.asarray(input_u, dtype=np.float32)
    input_v_arr = np.asarray(input_v, dtype=np.float32)

    fig, axes = plt.subplots(2, 6, figsize=(16.8, 6.0))
    col_titles = [
        "Input",
        "Clean Ground Truth",
        "Noisy Ground Truth",
        "Predicted",
        "Abs Error vs Clean GT",
        "Abs Error vs Noisy GT",
    ]
    rows = [
        (
            "u",
            input_u_arr,
            target_u_clean,
            target_u_noisy_arr,
            pred_u_arr,
            np.abs(pred_u_arr - target_u_clean),
            np.abs(pred_u_arr - target_u_noisy_arr),
        ),
        (
            "v",
            input_v_arr,
            target_v_clean,
            target_v_noisy_arr,
            pred_v_arr,
            np.abs(pred_v_arr - target_v_clean),
            np.abs(pred_v_arr - target_v_noisy_arr),
        ),
    ]

    for row_idx, (name, inp, tgt_clean, tgt_noisy, pred, err_clean, err_noisy) in enumerate(rows):
        field_norm = _shared_field_norm([inp, tgt_clean, tgt_noisy, pred], low_pct=1.0, high_pct=99.0)
        data_panels = [
            (inp, input_cmap, "field"),
            (tgt_clean, output_cmap, "field"),
            (tgt_noisy, output_cmap, "field"),
            (pred, output_cmap, "field"),
            (err_clean, "magma", "error"),
            (err_noisy, "magma", "error"),
        ]

        for col_idx, (data, panel_cmap, panel_kind) in enumerate(data_panels):
            ax = axes[row_idx, col_idx]
            if col_idx == 0:
                im = ax.imshow(data, cmap=panel_cmap, norm=field_norm)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(input_border_width)
                    spine.set_edgecolor(input_border_color)
            elif panel_kind == "field":
                im = ax.imshow(data, cmap=panel_cmap, norm=field_norm)
            else:
                im = ax.imshow(data, cmap=panel_cmap)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx])
            if col_idx == 0:
                ax.set_ylabel(f"{model_label} {name}", rotation=90, labelpad=8, fontsize=9)
            _annotate_min_max(ax, data)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    legend_handles = [
        _legend_patch(input_cmap, "Input colormap"),
        _legend_patch(output_cmap, "Field colormap"),
        _legend_patch("magma", "Error colormap"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=True,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.05, right=0.84, top=0.93, bottom=0.05, wspace=0.28, hspace=0.22)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_trajectory_error_rows(
    rows: List[Dict[str, np.ndarray]],
    step_indices: List[int],
    output_path: str | Path,
    title: str,
    cmap: str = "inferno",
    gamma: float = 0.55,
) -> None:
    """Save multi-row trajectory absolute-error heatmaps at selected timesteps."""
    if not rows:
        return
    if not step_indices:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(rows)
    n_cols = len(step_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 2.0 * n_rows), squeeze=False)
    labels = [str(row.get("label", f"row_{row_idx}")) for row_idx, row in enumerate(rows)]
    label_colors = _row_label_colors(labels)

    last_im = None
    for row_idx, row in enumerate(rows):
        pred = np.asarray(row["pred"])
        target = np.asarray(row["target"])
        label = str(row.get("label", f"row_{row_idx}"))
        max_step = pred.shape[0] - 1

        for col_idx, step in enumerate(step_indices):
            idx = int(np.clip(step, 0, max_step))
            err = np.abs(pred[idx] - target[idx])
            ax = axes[row_idx, col_idx]
            vmax_panel = float(np.percentile(err, 99.5))
            if not np.isfinite(vmax_panel) or vmax_panel <= 1e-12:
                vmax_panel = float(np.max(err))
            if not np.isfinite(vmax_panel) or vmax_panel <= 1e-12:
                vmax_panel = 1e-12
            norm = mcolors.PowerNorm(gamma=max(0.05, float(gamma)), vmin=0.0, vmax=vmax_panel)
            last_im = ax.imshow(err, cmap=cmap, norm=norm)
            ax.axis("off")
            _annotate_min_max(ax, err)

            if row_idx == 0:
                ax.set_title(f"t={idx}")
            if col_idx == 0:
                ax.set_ylabel(label, rotation=0, labelpad=54, va="center", color=label_colors[label], fontsize=8)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("Absolute error")

    legend_handles = [
        Line2D([0], [0], color=label_colors[label], lw=3, label=label)
        for label in _ordered_unique(labels)
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        frameon=True,
        fontsize=7,
    )

    fig.subplots_adjust(left=0.08, right=0.79, top=0.93, bottom=0.04, wspace=0.02, hspace=0.08)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_trajectory_field_rows(
    rows: List[Dict[str, np.ndarray]],
    step_indices: List[int],
    output_path: str | Path,
    title: str,
    cmap: str = "turbo",
) -> None:
    """Save multi-row trajectory field snapshots at selected timesteps."""
    if not rows:
        return
    if not step_indices:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(rows)
    n_cols = len(step_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 2.0 * n_rows), squeeze=False)
    labels = [str(row.get("label", f"row_{row_idx}")) for row_idx, row in enumerate(rows)]
    label_colors = _row_label_colors(labels)
    scale_frames: List[np.ndarray] = []
    for row in rows:
        traj = np.asarray(row["traj"])
        max_step = traj.shape[0] - 1
        for step in step_indices:
            idx = int(np.clip(step, 0, max_step))
            scale_frames.append(np.asarray(traj[idx], dtype=np.float32))
    shared_norm = _shared_field_norm(scale_frames, low_pct=1.0, high_pct=99.0)

    last_im = None
    for row_idx, row in enumerate(rows):
        traj = np.asarray(row["traj"])
        label = str(row.get("label", f"row_{row_idx}"))
        max_step = traj.shape[0] - 1

        for col_idx, step in enumerate(step_indices):
            idx = int(np.clip(step, 0, max_step))
            frame = traj[idx]
            ax = axes[row_idx, col_idx]
            last_im = ax.imshow(frame, cmap=cmap, norm=shared_norm)
            ax.axis("off")
            _annotate_min_max(ax, frame)

            if row_idx == 0:
                ax.set_title(f"t={idx}")
            if col_idx == 0:
                ax.set_ylabel(label, rotation=0, labelpad=56, va="center", color=label_colors[label], fontsize=8)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("Field value (shared scale)")

    legend_handles = [
        Line2D([0], [0], color=label_colors[label], lw=3, label=label)
        for label in _ordered_unique(labels)
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        frameon=True,
        fontsize=7,
    )

    fig.subplots_adjust(left=0.1, right=0.79, top=0.93, bottom=0.04, wspace=0.02, hspace=0.08)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _finite_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def save_clean_noisy_metric_bar(
    clean_value: float,
    noisy_value: float,
    metric_label: str,
    output_path: str | Path,
    title: str,
) -> None:
    """Save a 2-bar clean/noisy chart for one scalar diagnostic."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Clean", "Noisy"]
    values = [float(clean_value), float(noisy_value)]
    colors = ["#0072B2", "#D55E00"]

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    bars = ax.bar(labels, values, color=colors, width=0.65)
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ymax = max([v for v in values if np.isfinite(v)] + [1e-12])
    ax.set_ylim(0.0, ymax * 1.15)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                bar.get_height(),
                f"{value:.3e}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_spectral_band_error_plot(
    clean_band_error: List[float],
    noisy_band_error: List[float],
    band_labels: List[str],
    output_path: str | Path,
    title: str,
    band_centers: List[float] | None = None,
) -> None:
    """Save clean/noisy spectral band-error profile."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clean = np.asarray(clean_band_error, dtype=np.float64)
    noisy = np.asarray(noisy_band_error, dtype=np.float64)
    n_bands = min(clean.size, noisy.size, len(band_labels))
    if n_bands <= 0:
        return

    clean = clean[:n_bands]
    noisy = noisy[:n_bands]
    labels = [str(lbl) for lbl in band_labels[:n_bands]]

    x = np.arange(1, n_bands + 1, dtype=np.float64)
    xlabel = "Spectral band index"
    if band_centers is not None:
        centers = np.asarray(band_centers[:n_bands], dtype=np.float64)
        if centers.size == n_bands and np.all(np.isfinite(centers)):
            x = centers
            xlabel = "Band center wavenumber"

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(x, clean, marker="o", color="#0072B2", label="Clean")
    ax.plot(x, noisy, marker="o", color="#D55E00", label="Noisy")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Abs band-fraction error")
    ax.set_title(title)
    if xlabel == "Spectral band index":
        ax.set_xticks(np.arange(1, n_bands + 1, dtype=np.float64))
        ax.set_xticklabels(labels, rotation=0)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_metric_vs_l2_grid(
    l2_clean: List[float],
    l2_noisy: List[float],
    metric_series: Mapping[str, Mapping[str, List[float]]],
    output_path: str | Path,
    title: str,
) -> None:
    """Save metric-vs-L2 scatter grid for clean/noisy trajectories."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not metric_series:
        return

    items = list(metric_series.items())
    n_rows = len(items)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11.0, max(3.0, 3.1 * n_rows)), squeeze=False)

    l2_clean_arr = np.asarray(l2_clean, dtype=np.float64)
    l2_noisy_arr = np.asarray(l2_noisy, dtype=np.float64)

    for row_idx, (metric_name, payload) in enumerate(items):
        clean_vals = np.asarray(payload.get("clean", []), dtype=np.float64)
        noisy_vals = np.asarray(payload.get("noisy", []), dtype=np.float64)
        panels = [
            (axes[row_idx, 0], l2_clean_arr, clean_vals, "Clean", "#0072B2"),
            (axes[row_idx, 1], l2_noisy_arr, noisy_vals, "Noisy", "#D55E00"),
        ]
        for ax, x_raw, y_raw, label, color in panels:
            x, y = _finite_pairs(x_raw, y_raw)
            if x.size > 0:
                ax.scatter(x, y, s=24, alpha=0.8, color=color, edgecolors="none")
            if x.size >= 2:
                x_center = x - np.mean(x)
                y_center = y - np.mean(y)
                denom = float(np.sqrt(np.sum(x_center * x_center) * np.sum(y_center * y_center)))
                corr = float(np.sum(x_center * y_center) / denom) if denom > 1e-12 else float("nan")
                if np.isfinite(corr):
                    coeff = np.polyfit(x, y, deg=1)
                    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                    y_line = coeff[0] * x_line + coeff[1]
                    ax.plot(x_line, y_line, color=color, linewidth=1.2, alpha=0.85)
                    ax.set_title(f"{metric_name} vs L2 ({label})  r={corr:.2f}", fontsize=9)
                else:
                    ax.set_title(f"{metric_name} vs L2 ({label})", fontsize=9)
            else:
                ax.set_title(f"{metric_name} vs L2 ({label})", fontsize=9)

            ax.set_xlabel("L2")
            ax.set_ylabel(metric_name)
            ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(path, dpi=200)
    plt.close(fig)
