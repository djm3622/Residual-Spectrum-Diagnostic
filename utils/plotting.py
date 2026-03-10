"""Lightweight plotting helpers for single-seed runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


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
