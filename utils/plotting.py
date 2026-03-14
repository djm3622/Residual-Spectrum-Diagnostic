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

PUB_DPI = 300
METHOD_COLORS = {
    "clean": "#0072B2",
    "noisy": "#D55E00",
}
REFERENCE_COLORS = {
    "clean_gt": "#1B9E77",
    "noisy_gt": "#E6AB02",
}


def _set_pub_rcparams() -> None:
    """Apply publication defaults consistently across all generated figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


_set_pub_rcparams()


def _color_for_label(label: str) -> str:
    text = str(label).lower()
    if "clean pred" in text:
        return METHOD_COLORS["clean"]
    if "noisy pred" in text:
        return METHOD_COLORS["noisy"]
    if "clean gt" in text:
        return REFERENCE_COLORS["clean_gt"]
    if "noisy gt" in text:
        return REFERENCE_COLORS["noisy_gt"]
    if "noisy" in text and "clean" not in text:
        return METHOD_COLORS["noisy"]
    if "clean" in text and "noisy" not in text:
        return METHOD_COLORS["clean"]
    return "#444444"


def _add_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    handles: List[Line2D | Patch],
    *,
    is_multi_panel: bool,
    single_loc: str = "upper right",
    single_ncol: int = 1,
    multi_ncol: int | None = None,
) -> None:
    if is_multi_panel:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=max(1, int(multi_ncol if multi_ncol is not None else len(handles))),
            frameon=False,
        )
        return

    ax.legend(
        handles=handles,
        loc=single_loc,
        ncol=max(1, int(single_ncol)),
        frameon=False,
    )


def _style_table_cell(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#222222")


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


def _row_label_colors(labels: List[str]) -> Dict[str, str]:
    unique_labels = _ordered_unique(labels)
    return {
        label: _color_for_label(label)
        for label in unique_labels
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
    _ = title
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["L2", "HFV", "LFV"]
    clean_values = [metrics["clean_l2"], metrics["clean_hfv"], metrics["clean_lfv"]]
    noisy_values = [metrics["noisy_l2"], metrics["noisy_hfv"], metrics["noisy_lfv"]]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar([i - width / 2 for i in x], clean_values, width=width, label="Clean", color=METHOD_COLORS["clean"])
    ax.bar([i + width / 2 for i in x], noisy_values, width=width, label="Noisy", color=METHOD_COLORS["noisy"])

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric value")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7, axis="y")
    _add_legend(
        fig,
        ax,
        [
            Patch(facecolor=METHOD_COLORS["clean"], edgecolor="none", label="Clean"),
            Patch(facecolor=METHOD_COLORS["noisy"], edgecolor="none", label="Noisy"),
        ],
        is_multi_panel=False,
        single_loc="upper right",
        single_ncol=2,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=PUB_DPI, bbox_inches="tight")
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
    _ = title
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
        (inp, input_cmap, "field"),
        (clean_target, cmap, "field"),
        (noisy_target, cmap, "field"),
        (pred, cmap, "field"),
        (err_vs_clean, "magma", "error"),
        (err_vs_noisy, "magma", "error"),
    ]

    for idx, (data, panel_cmap, panel_kind) in enumerate(panels):
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
    _add_legend(
        fig,
        axes[0],
        legend_handles,
        is_multi_panel=True,
        multi_ncol=len(legend_handles),
    )

    fig.subplots_adjust(left=0.05, right=0.95, top=0.86, bottom=0.08, wspace=0.28)
    fig.savefig(path, dpi=PUB_DPI, bbox_inches="tight")
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
    _ = title
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
    _add_legend(
        fig,
        axes[0, 0],
        legend_handles,
        is_multi_panel=True,
        multi_ncol=len(legend_handles),
    )
    fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.05, wspace=0.28, hspace=0.22)
    fig.savefig(path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)


def save_trajectory_error_rows(
    rows: List[Dict[str, np.ndarray]],
    step_indices: List[int],
    output_path: str | Path,
    title: str,
    cmap: str = "magma",
    gamma: float = 0.55,
) -> None:
    """Save multi-row trajectory absolute-error heatmaps at selected timesteps."""
    _ = title
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
            _style_table_cell(ax)
            _annotate_min_max(ax, err)

            if row_idx == 0:
                ax.set_xlabel(f"t = {int(step)}")
                ax.xaxis.set_label_position("top")
            if col_idx == 0:
                ax.set_ylabel(
                    label,
                    rotation=0,
                    labelpad=58,
                    ha="right",
                    va="center",
                    color=label_colors[label],
                    fontsize=9,
                )

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("Absolute error")

    fig.subplots_adjust(left=0.2, right=0.94, top=0.9, bottom=0.04, wspace=0.02, hspace=0.08)
    fig.savefig(path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)


def save_trajectory_field_rows(
    rows: List[Dict[str, np.ndarray]],
    step_indices: List[int],
    output_path: str | Path,
    title: str,
    cmap: str = "viridis",
) -> None:
    """Save multi-row trajectory field snapshots at selected timesteps."""
    _ = title
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
            _style_table_cell(ax)
            _annotate_min_max(ax, frame)

            if row_idx == 0:
                ax.set_xlabel(f"t = {int(step)}")
                ax.xaxis.set_label_position("top")
            if col_idx == 0:
                ax.set_ylabel(
                    label,
                    rotation=0,
                    labelpad=58,
                    ha="right",
                    va="center",
                    color=label_colors[label],
                    fontsize=9,
                )

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("Field value (shared scale)")

    fig.subplots_adjust(left=0.2, right=0.94, top=0.9, bottom=0.04, wspace=0.02, hspace=0.08)
    fig.savefig(path, dpi=PUB_DPI, bbox_inches="tight")
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
    _ = title
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Clean", "Noisy"]
    values = [float(clean_value), float(noisy_value)]
    colors = [METHOD_COLORS["clean"], METHOD_COLORS["noisy"]]

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    bars = ax.bar(labels, values, color=colors, width=0.65)
    ax.set_ylabel(metric_label)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7, axis="y")
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
    fig.savefig(path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)


def _band_axis_values(
    n_bands: int,
    band_labels: List[str],
    band_centers: List[float] | None = None,
) -> tuple[np.ndarray, str, List[str]]:
    labels = [str(lbl) for lbl in band_labels[:n_bands]]
    x = np.arange(1, n_bands + 1, dtype=np.float64)
    xlabel = "Spectral band index"
    if band_centers is not None:
        centers = np.asarray(band_centers[:n_bands], dtype=np.float64)
        if centers.size == n_bands and np.all(np.isfinite(centers)):
            x = centers
            xlabel = "Band center wavenumber"
    return x, xlabel, labels


def save_band_profile_plot(
    clean_band_values: List[float],
    noisy_band_values: List[float],
    band_labels: List[str],
    output_path: str | Path,
    title: str,
    y_label: str,
    band_centers: List[float] | None = None,
) -> None:
    """Save clean/noisy per-band profile for any bandwise metric."""
    _ = title
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clean = np.asarray(clean_band_values, dtype=np.float64)
    noisy = np.asarray(noisy_band_values, dtype=np.float64)
    n_bands = min(clean.size, noisy.size, len(band_labels))
    if n_bands <= 0:
        return

    clean = clean[:n_bands]
    noisy = noisy[:n_bands]
    x, xlabel, labels = _band_axis_values(n_bands, band_labels, band_centers)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(x, clean, marker="o", color=METHOD_COLORS["clean"], label="Clean")
    ax.plot(x, noisy, marker="o", color=METHOD_COLORS["noisy"], label="Noisy")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_label)
    if xlabel == "Spectral band index":
        ax.set_xticks(np.arange(1, n_bands + 1, dtype=np.float64))
        ax.set_xticklabels(labels, rotation=0)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    _add_legend(
        fig,
        ax,
        [
            Line2D([0], [0], color=METHOD_COLORS["clean"], marker="o", label="Clean"),
            Line2D([0], [0], color=METHOD_COLORS["noisy"], marker="o", label="Noisy"),
        ],
        is_multi_panel=False,
        single_loc="upper right",
        single_ncol=2,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=PUB_DPI, bbox_inches="tight")
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
    save_band_profile_plot(
        clean_band_values=clean_band_error,
        noisy_band_values=noisy_band_error,
        band_labels=band_labels,
        output_path=output_path,
        title=title,
        y_label="Abs band-fraction error",
        band_centers=band_centers,
    )


def _bootstrap_gap_arrays(
    bootstrap_rows: List[Mapping[str, float]],
    n_bands: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.full((n_bands,), np.nan, dtype=np.float64)
    low = np.full((n_bands,), np.nan, dtype=np.float64)
    high = np.full((n_bands,), np.nan, dtype=np.float64)
    for idx in range(min(n_bands, len(bootstrap_rows))):
        row = bootstrap_rows[idx]
        mean[idx] = float(row.get("gap_mean", np.nan))
        low[idx] = float(row.get("ci_low", np.nan))
        high[idx] = float(row.get("ci_high", np.nan))
    return mean, low, high


def save_dual_band_gap_bootstrap_plot(
    band_labels: List[str],
    fraction_gap_bootstrap: List[Mapping[str, float]],
    coeff_mse_gap_bootstrap: List[Mapping[str, float]],
    output_path: str | Path,
    title: str,
    band_centers: List[float] | None = None,
) -> None:
    """Save 95% bootstrap CIs for noisy-clean spectral gaps (fractional + coeff-MSE)."""
    _ = title
    n_bands = len(band_labels)
    if n_bands <= 0:
        return

    frac_mean, frac_low, frac_high = _bootstrap_gap_arrays(fraction_gap_bootstrap, n_bands)
    coeff_mean, coeff_low, coeff_high = _bootstrap_gap_arrays(coeff_mse_gap_bootstrap, n_bands)
    if not (
        np.any(np.isfinite(frac_mean))
        or np.any(np.isfinite(frac_low))
        or np.any(np.isfinite(frac_high))
        or np.any(np.isfinite(coeff_mean))
        or np.any(np.isfinite(coeff_low))
        or np.any(np.isfinite(coeff_high))
    ):
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x, xlabel, labels = _band_axis_values(n_bands, band_labels, band_centers)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), squeeze=False)
    panels = [
        (
            axes[0, 0],
            frac_mean,
            frac_low,
            frac_high,
            "Band-fraction gap (noisy-clean)",
            "#7B3294",
        ),
        (
            axes[0, 1],
            coeff_mean,
            coeff_low,
            coeff_high,
            "Fourier coeff-MSE gap (noisy-clean)",
            "#008837",
        ),
    ]

    for ax, mean, low, high, ylabel, color in panels:
        finite_mask = np.isfinite(mean) & np.isfinite(low) & np.isfinite(high)
        if np.any(finite_mask):
            ax.fill_between(x[finite_mask], low[finite_mask], high[finite_mask], color=color, alpha=0.22)
            ax.plot(x[finite_mask], mean[finite_mask], color=color, marker="o", linewidth=1.4)
        ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlabel == "Spectral band index":
            ax.set_xticks(np.arange(1, n_bands + 1, dtype=np.float64))
            ax.set_xticklabels(labels, rotation=0)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)


def save_metric_vs_l2_grid(
    l2_clean: List[float],
    l2_noisy: List[float],
    metric_series: Mapping[str, Mapping[str, List[float]]],
    output_path: str | Path,
    title: str,
) -> None:
    """Save metric-vs-L2 scatter grid for clean/noisy trajectories."""
    _ = title
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
            (axes[row_idx, 0], l2_clean_arr, clean_vals, "Clean", METHOD_COLORS["clean"]),
            (axes[row_idx, 1], l2_noisy_arr, noisy_vals, "Noisy", METHOD_COLORS["noisy"]),
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
                    ax.text(
                        0.02,
                        0.97,
                        f"r={corr:.2f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="#222222",
                        bbox={"facecolor": "white", "alpha": 0.75, "pad": 1.8, "edgecolor": "none"},
                    )

            ax.set_xlabel("L2")
            ax.set_ylabel(f"{metric_name} ({label})")
            ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    _add_legend(
        fig,
        axes[0, 0],
        [
            Line2D([0], [0], marker="o", linestyle="-", color=METHOD_COLORS["clean"], label="Clean"),
            Line2D([0], [0], marker="o", linestyle="-", color=METHOD_COLORS["noisy"], label="Noisy"),
        ],
        is_multi_panel=True,
        multi_ncol=2,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)
