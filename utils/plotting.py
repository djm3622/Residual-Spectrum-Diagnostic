"""Lightweight plotting helpers for single-seed runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# Use writable cache location in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
        vmax = float(max(np.max(inp), np.max(tgt), np.max(pred), 1e-12))
        data_panels = [
            (inp, f"{name}: Input"),
            (tgt, f"{name}: Target (y)"),
            (pred, f"{name}: Predicted"),
            (err, f"{name}: Absolute Error"),
        ]

        for col_idx, (data, panel_title) in enumerate(data_panels):
            ax = axes[row_idx, col_idx]
            if col_idx == 0:
                im = ax.imshow(data, cmap=input_cmap, vmin=0.0, vmax=vmax)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(input_border_width)
                    spine.set_edgecolor(input_border_color)
            elif col_idx < 3:
                im = ax.imshow(data, cmap=output_cmap, vmin=0.0, vmax=vmax)
            else:
                im = ax.imshow(data, cmap="magma")
            ax.set_title(panel_title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    rel_l2_u = np.linalg.norm(pred_u - target_u) / (np.linalg.norm(target_u) + 1e-12)
    rel_l2_v = np.linalg.norm(pred_v - target_v) / (np.linalg.norm(target_v) + 1e-12)
    fig.suptitle(f"{title} | rel-L2(u)={rel_l2_u:.4f}, rel-L2(v)={rel_l2_v:.4f}", fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
