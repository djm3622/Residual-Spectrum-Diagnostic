#!/usr/bin/env python3
"""Inspect and visualize full trajectories from FNO Navier-Stokes MAT dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import h5py
from PIL import Image
from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="visualize_fno_ns_mat.py",
        description="Compute stats and save lightweight trajectory previews for FNO NS MAT files.",
    )
    parser.add_argument(
        "--mat-file",
        type=str,
        default="external_data/fno/NavierStokes_V1e-3_N5000_T50.mat",
        help="Path to MAT dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="external_data/fno/analysis",
        help="Directory for summary JSON and preview GIF/PNGs.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Trajectory index to visualize.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=25,
        help="Maximum number of frames in GIF preview.",
    )
    return parser.parse_args()


def _to_rgb(frame: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = (vmax - vmin) if vmax > vmin else 1.0
    z = np.clip((frame - vmin) / denom, 0.0, 1.0)
    r = (255.0 * z).astype(np.uint8)
    b = (255.0 * (1.0 - z)).astype(np.uint8)
    g = (255.0 * (1.0 - np.abs(2.0 * z - 1.0))).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _write_panel(frames: list[np.ndarray], out_path: Path) -> None:
    vmin = float(min(np.min(f) for f in frames))
    vmax = float(max(np.max(f) for f in frames))
    imgs = [Image.fromarray(_to_rgb(f, vmin, vmax)) for f in frames]
    w = sum(i.size[0] for i in imgs)
    h = max(i.size[1] for i in imgs)
    panel = Image.new("RGB", (w, h))
    x = 0
    for img in imgs:
        panel.paste(img, (x, 0))
        x += img.size[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def _write_gif(frames: list[np.ndarray], out_path: Path) -> None:
    if len(frames) < 2:
        return
    vmin = float(min(np.min(f) for f in frames))
    vmax = float(max(np.max(f) for f in frames))
    imgs = [Image.fromarray(_to_rgb(f, vmin, vmax)) for f in frames]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=200, loop=0)


def _stats_h5_u(u_ds: h5py.Dataset, chunk_n: int = 64) -> Dict[str, float]:
    # u_ds shape for this dataset: [T, H, W, N]
    t, h, w, n = map(int, u_ds.shape)
    total_count = 0
    s1 = 0.0
    s2 = 0.0
    vmin = np.inf
    vmax = -np.inf
    for i in range(0, n, chunk_n):
        j = min(n, i + chunk_n)
        block = np.asarray(u_ds[:, :, :, i:j], dtype=np.float32)
        vmin = min(vmin, float(np.min(block)))
        vmax = max(vmax, float(np.max(block)))
        s1 += float(np.sum(block, dtype=np.float64))
        s2 += float(np.sum(block * block, dtype=np.float64))
        total_count += int(block.size)
    mean = s1 / total_count
    var = max(0.0, s2 / total_count - mean * mean)
    return {"min": float(vmin), "max": float(vmax), "mean": float(mean), "std": float(np.sqrt(var))}


def main() -> None:
    args = parse_args()
    mat_file = Path(args.mat_file).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mat_file.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_file}")

    keys: list[str] = []
    n = h = w = ts = 0
    idx = 0
    a_shape = None
    t_shape = None
    u_stats = None

    # Try scipy first; fallback to h5py for matlab v7.3.
    try:
        data = loadmat(mat_file)
        keys = sorted(k for k in data.keys() if not k.startswith("__"))
        if "u" not in data:
            raise KeyError(f"'u' not found in MAT keys {keys}")
        # Common scipy orientation for these files: [N,H,W,T]
        u = np.asarray(data["u"], dtype=np.float32)
        if u.ndim != 4:
            raise ValueError(f"Expected u shape [N,H,W,T], got {u.shape}")
        n, h, w, ts = map(int, u.shape)
        idx = max(0, min(int(args.sample_index), n - 1))
        traj = np.moveaxis(u[idx], -1, 0)  # [T,H,W]
        u_stats = {
            "min": float(np.min(u)),
            "max": float(np.max(u)),
            "mean": float(np.mean(u)),
            "std": float(np.std(u)),
        }
        a = np.asarray(data["a"], dtype=np.float32) if "a" in data else None
        t = np.asarray(data["t"], dtype=np.float32).reshape(-1) if "t" in data else None
        a_shape = list(a.shape) if a is not None else None
        t_shape = list(t.shape) if t is not None else None
    except NotImplementedError:
        with h5py.File(mat_file, "r") as f:
            keys = sorted(list(f.keys()))
            if "u" not in f:
                raise KeyError(f"'u' not found in MAT keys {keys}")
            # For this dataset u is stored as [T,H,W,N].
            u_ds = f["u"]
            if u_ds.ndim != 4:
                raise ValueError(f"Expected u dataset rank 4, got shape {u_ds.shape}")
            ts, h, w, n = map(int, u_ds.shape)
            idx = max(0, min(int(args.sample_index), n - 1))
            traj = np.asarray(u_ds[:, :, :, idx], dtype=np.float32)  # [T,H,W]
            u_stats = _stats_h5_u(u_ds, chunk_n=64)
            a_shape = list(f["a"].shape) if "a" in f else None
            t_shape = list(f["t"].shape) if "t" in f else None

    stride = max(1, int(np.ceil(ts / max(1, int(args.max_frames)))))
    frames = [traj[i] for i in range(0, ts, stride)]
    panel_frames = [traj[0], traj[ts // 2], traj[ts - 1]]

    panel_path = out_dir / "trajectory_panel.png"
    gif_path = out_dir / "trajectory_preview.gif"
    _write_panel(panel_frames, panel_path)
    _write_gif(frames, gif_path)

    summary: Dict[str, object] = {
        "mat_file": str(mat_file),
        "keys": keys,
        "u_shape": [n, h, w, ts],
        "trajectory_count": n,
        "grid": [h, w],
        "time_steps": ts,
        "selected_sample_index": idx,
        "u_stats": u_stats,
        "a_shape": a_shape,
        "t_shape": t_shape,
        "outputs": {
            "panel_png": str(panel_path),
            "preview_gif": str(gif_path) if gif_path.exists() else None,
        },
    }

    summary_path = out_dir / "stats_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] wrote {summary_path}")
    print(f"[ok] wrote {panel_path}")
    if gif_path.exists():
        print(f"[ok] wrote {gif_path}")


if __name__ == "__main__":
    main()
