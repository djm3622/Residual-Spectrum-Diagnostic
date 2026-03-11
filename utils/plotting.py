"""Lightweight plotting helpers for single-seed runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# Use writable cache location in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
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
) -> None:
    """Save a 4-panel scalar fit diagnostic: input, target, pred, absolute error."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    error = np.abs(pred_field - target_field)
    vmax = float(max(np.max(np.abs(input_field)), np.max(np.abs(target_field)), np.max(np.abs(pred_field)), 1e-12))

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    panels = [
        (input_field, "Input"),
        (target_field, "Target (y)"),
        (pred_field, "Predicted"),
        (error, "Absolute Error"),
    ]

    for idx, (data, name) in enumerate(panels):
        ax = axes[idx]
        if idx == 0:
            # Keep input visually distinct from output-related panels.
            im = ax.imshow(data, cmap=input_cmap, vmin=-vmax, vmax=vmax)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(input_border_width)
                spine.set_edgecolor(input_border_color)
        elif idx < 3:
            im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, cmap="magma")
        ax.set_title(name)
        _annotate_min_max(ax, data)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    rel_l2 = np.linalg.norm(pred_field - target_field) / (np.linalg.norm(target_field) + 1e-12)
    fig.suptitle(f"{title} | rel-L2={rel_l2:.4f}", fontsize=11)
    plt.tight_layout()
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
) -> None:
    """Save coupled fit diagnostics for (u, v): 2 rows x 4 columns."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    err_u = np.abs(pred_u - target_u)
    err_v = np.abs(pred_v - target_v)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    rows = [
        ("u", input_u, target_u, pred_u, err_u),
        ("v", input_v, target_v, pred_v, err_v),
    ]

    for row_idx, (name, inp, tgt, pred, err) in enumerate(rows):
        field_norm = _shared_field_norm([inp, tgt, pred], low_pct=1.0, high_pct=99.0)
        data_panels = [
            (inp, f"{name}: Input"),
            (tgt, f"{name}: Target (y)"),
            (pred, f"{name}: Predicted"),
            (err, f"{name}: Absolute Error"),
        ]

        for col_idx, (data, panel_title) in enumerate(data_panels):
            ax = axes[row_idx, col_idx]
            if col_idx == 0:
                im = ax.imshow(data, cmap=input_cmap, norm=field_norm)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(input_border_width)
                    spine.set_edgecolor(input_border_color)
            elif col_idx < 3:
                im = ax.imshow(data, cmap=output_cmap, norm=field_norm)
            else:
                im = ax.imshow(data, cmap="magma")
            ax.set_title(panel_title)
            _annotate_min_max(ax, data)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    rel_l2_u = np.linalg.norm(pred_u - target_u) / (np.linalg.norm(target_u) + 1e-12)
    rel_l2_v = np.linalg.norm(pred_v - target_v) / (np.linalg.norm(target_v) + 1e-12)
    fig.suptitle(
        f"{title} | one-step rel-L2(u)={rel_l2_u:.4f}, one-step rel-L2(v)={rel_l2_v:.4f}",
        fontsize=11,
    )
    plt.tight_layout()
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
                ax.set_ylabel(label, rotation=0, labelpad=52, va="center")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        cbar.set_label("Absolute error")

    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.93, top=0.9, bottom=0.04, wspace=0.02, hspace=0.08)
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
                ax.set_ylabel(label, rotation=0, labelpad=56, va="center")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        cbar.set_label("Field value (shared scale)")

    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.1, right=0.93, top=0.9, bottom=0.04, wspace=0.02, hspace=0.08)
    fig.savefig(path, dpi=200)
    plt.close(fig)
