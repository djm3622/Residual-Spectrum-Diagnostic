#!/usr/bin/env python3
"""Download PDEBench data files and optionally patch local run config."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import yaml

PDEBENCH_METADATA_URL = (
    "https://raw.githubusercontent.com/pdebench/PDEBench/main/"
    "pdebench/data_download/pdebench_data_urls.csv"
)

PDE_NAMES = (
    "advection",
    "burgers",
    "1d_cfd",
    "diff_sorp",
    "1d_reacdiff",
    "2d_cfd",
    "darcy",
    "2d_reacdiff",
    "ns_incom",
    "swe",
    "3d_cfd",
)

PDE_NAME_ALIASES = {
    "reaction_diffusion": "2d_reacdiff",
    "reaction-diffusion": "2d_reacdiff",
    "rd": "2d_reacdiff",
    "gray_scott": "2d_reacdiff",
    "grayscott": "2d_reacdiff",
}

REQUIRED_METADATA_COLUMNS = ("PDE", "URL", "Path", "Filename", "MD5")
H5_NUMERIC_KINEMATIC_KEYS = (
    "nu",
    "viscosity",
    "kinematic_viscosity",
)
RD_PDE_NAMES = {"1d_reacdiff", "2d_reacdiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_pdebench.py",
        description=(
            "Download PDEBench files into a standard local directory and "
            "auto-fill data.external.pdebench.file_path in a YAML config."
        ),
    )
    parser.add_argument(
        "--pde-name",
        action="append",
        dest="pde_names",
        required=True,
        help=(
            "PDEBench dataset key to download. Repeat for multiple values. "
            f"Choices: {', '.join(PDE_NAMES)}. "
            "Reaction-diffusion aliases: reaction_diffusion, reaction-diffusion, rd, gray_scott."
        ),
    )
    parser.add_argument(
        "--root-folder",
        type=str,
        default="external_data/pdebench",
        help="Root destination for downloaded PDEBench files (default: external_data/pdebench).",
    )
    parser.add_argument(
        "--config-yaml",
        type=str,
        default="configs/navier_stokes.yaml",
        help="Config YAML to patch (default: configs/navier_stokes.yaml).",
    )
    parser.add_argument(
        "--metadata-url",
        type=str,
        default=PDEBENCH_METADATA_URL,
        help="Override metadata CSV URL (defaults to official PDEBench GitHub raw URL).",
    )
    parser.add_argument(
        "--filename-contains",
        action="append",
        default=None,
        help="Optional filename substring filter. Repeat to allow multiple substrings.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1,
        help=(
            "Maximum number of files to download after filtering (default: 1). "
            "Use 0 for no limit."
        ),
    )
    parser.add_argument(
        "--autofill-source",
        action="store_true",
        default=True,
        help="Set data.external.source to 'pdebench' when patching config (default: true).",
    )
    parser.add_argument(
        "--no-autofill-source",
        action="store_true",
        default=False,
        help="Do not set data.external.source when patching config.",
    )
    parser.add_argument(
        "--align-run-config",
        action="store_true",
        default=True,
        help=(
            "Auto-align run config fields from downloaded HDF5 metadata "
            "(dataset_key/layout/dt, and physics forcing defaults) (default: true)."
        ),
    )
    parser.add_argument(
        "--no-align-run-config",
        action="store_true",
        default=False,
        help="Disable automatic run-config alignment.",
    )
    parser.add_argument(
        "--skip-config-update",
        action="store_true",
        default=False,
        help="Do not patch config YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print actions without downloading or writing config.",
    )
    return parser.parse_args()


def load_metadata_rows(metadata_url: str) -> List[Dict[str, str]]:
    try:
        with urlopen(metadata_url) as response:
            payload = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Failed to fetch metadata CSV ({metadata_url}): HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to fetch metadata CSV ({metadata_url}): {exc.reason}") from exc

    text = payload.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise RuntimeError("Metadata CSV is empty.")

    missing = [col for col in REQUIRED_METADATA_COLUMNS if col not in reader.fieldnames]
    if missing:
        raise RuntimeError(
            "Metadata CSV missing required columns: "
            f"{', '.join(missing)}. Found: {reader.fieldnames}"
        )
    return rows


def filter_rows(
    rows: List[Dict[str, str]],
    pde_names: List[str],
    filename_contains: List[str] | None,
    max_files: int,
) -> List[Dict[str, str]]:
    allowed = {
        PDE_NAME_ALIASES.get(name.strip().lower(), name.strip().lower())
        for name in pde_names
    }
    unsupported = [name for name in sorted(allowed) if name not in PDE_NAMES]
    if unsupported:
        raise ValueError(
            "Unsupported --pde-name values: "
            f"{', '.join(unsupported)}. Choices: {', '.join(PDE_NAMES)}"
        )

    selected = [
        row
        for row in rows
        if str(row["PDE"]).strip().lower() in allowed
    ]

    if filename_contains:
        filters = [frag for frag in (item.strip() for item in filename_contains) if frag]
        if filters:
            selected = [
                row
                for row in selected
                if any(fragment in str(row["Filename"]) for fragment in filters)
            ]

    if max_files > 0:
        selected = selected[:max_files]

    return selected


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_row(root_folder: Path, row: Dict[str, str], dry_run: bool) -> Path:
    rel_folder = Path(str(row["Path"]).strip())
    filename = str(row["Filename"]).strip()
    url = str(row["URL"]).strip()
    expected_md5 = str(row.get("MD5", "")).strip().lower()

    destination_dir = root_folder / rel_folder
    destination_file = destination_dir / filename

    if dry_run:
        print(f"[dry-run] download {url} -> {destination_file}")
        return destination_file

    destination_dir.mkdir(parents=True, exist_ok=True)

    if destination_file.exists() and expected_md5:
        existing_md5 = md5sum(destination_file)
        if existing_md5 == expected_md5:
            print(f"[skip] {destination_file} (md5 ok)")
            return destination_file
        print(f"[warn] {destination_file} md5 mismatch, re-downloading")

    tmp_file = destination_file.with_suffix(destination_file.suffix + ".part")
    try:
        with urlopen(url) as response, tmp_file.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        tmp_file.replace(destination_file)
    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise

    if expected_md5:
        downloaded_md5 = md5sum(destination_file)
        if downloaded_md5 != expected_md5:
            raise RuntimeError(
                f"MD5 mismatch for {destination_file}. expected={expected_md5}, got={downloaded_md5}"
            )

    print(f"[ok] {destination_file}")
    return destination_file


def _float_from_any(value: Any) -> float | None:
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore[assignment]

    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None

    if np is not None:
        arr = np.asarray(value)
        if arr.size == 1:
            item = arr.reshape(-1)[0]
            try:
                return float(item)
            except Exception:
                return None

    try:
        return float(value)
    except Exception:
        return None


def _infer_layout_from_shape(shape: Tuple[int, ...]) -> str:
    if len(shape) == 3:
        return "THW"
    if len(shape) == 4:
        # Per-sample grouped RD data is commonly [T, H, W, C].
        if int(shape[-1]) <= 16:
            return "THWC"
        return "NTHW"
    if len(shape) == 5:
        # Prefer [batch, time, x, y, channels] for PDEBench NS velocity.
        if int(shape[-1]) <= 16:
            return "NTHWC"
        if int(shape[1]) <= 16:
            return "NCTHW"
        return "NTHWC"
    return "AUTO"


def _sample_group_keys(handle: Any) -> List[str]:
    try:
        import h5py
    except Exception:
        return []
    keys = [key for key in handle.keys() if isinstance(handle.get(key), h5py.Group)]
    return sorted(keys)


def _extract_rd_yaml_config(raw_value: Any) -> Dict[str, Any]:
    if raw_value is None:
        return {}
    if isinstance(raw_value, (bytes, bytearray)):
        try:
            raw_value = raw_value.decode("utf-8")
        except Exception:
            return {}
    if not isinstance(raw_value, str):
        return {}
    try:
        payload = yaml.safe_load(raw_value)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _extract_rd_physics_from_config_blob(config_blob: Dict[str, Any]) -> Dict[str, float]:
    sim = config_blob.get("sim", {})
    if not isinstance(sim, dict):
        return {}
    out: Dict[str, float] = {}
    for key, aliases in {
        "Du": ("Du", "du", "diffusion_u", "D_u"),
        "Dv": ("Dv", "dv", "diffusion_v", "D_v"),
        "F": ("F", "f", "feed", "feed_rate"),
        "k": ("k", "K", "kill", "kill_rate"),
    }.items():
        for alias in aliases:
            if alias in sim:
                value = _float_from_any(sim.get(alias))
                if value is not None:
                    out[key] = float(value)
                    break
    return out


def _extract_rd_domain_from_config_blob(config_blob: Dict[str, Any]) -> Dict[str, float]:
    sim = config_blob.get("sim", {})
    if not isinstance(sim, dict):
        return {}
    out: Dict[str, float] = {}
    x_left = _float_from_any(sim.get("x_left"))
    x_right = _float_from_any(sim.get("x_right"))
    y_bottom = _float_from_any(sim.get("y_bottom"))
    y_top = _float_from_any(sim.get("y_top"))
    t_final = _float_from_any(sim.get("t"))
    if x_left is not None and x_right is not None and x_right > x_left:
        out["Lx"] = float(x_right - x_left)
    if y_bottom is not None and y_top is not None and y_top > y_bottom:
        out["Ly"] = float(y_top - y_bottom)
    if t_final is not None and t_final > 0.0:
        out["t_final"] = float(t_final)
    return out


def inspect_h5_for_alignment(file_path: Path) -> Dict[str, Any]:
    """Inspect downloaded HDF5 and infer loader/config alignment values."""
    try:
        import h5py
        import numpy as np
    except Exception as exc:
        raise RuntimeError("Need h5py and numpy to inspect downloaded HDF5 metadata.") from exc

    if not file_path.exists():
        raise FileNotFoundError(f"Downloaded file not found for inspection: {file_path}")

    with h5py.File(file_path, "r") as handle:
        root_keys = list(handle.keys())
        dataset_candidates: Dict[str, Tuple[int, ...]] = {}

        def _collect_dataset(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                dataset_candidates[name] = tuple(int(dim) for dim in obj.shape)

        handle.visititems(_collect_dataset)

        sample_groups = _sample_group_keys(handle)
        grouped_candidate = None
        for group_key in sample_groups[: min(8, len(sample_groups))]:
            data_path = f"{group_key}/data"
            shape = dataset_candidates.get(data_path)
            if shape is not None and len(shape) >= 3:
                grouped_candidate = (group_key, "data", shape)
                break

        sample_grouped = grouped_candidate is not None
        sample_group_key = grouped_candidate[0] if grouped_candidate else None
        if grouped_candidate:
            chosen_key = grouped_candidate[1]
            chosen_path = f"{sample_group_key}/{chosen_key}"
            shape = grouped_candidate[2]
        else:
            preferred_keys = ("tensor", "velocity", "data")
            chosen_path = None
            for key in preferred_keys:
                shape = dataset_candidates.get(key)
                if shape is not None and len(shape) >= 4:
                    chosen_path = key
                    break
            if chosen_path is None:
                for key, shape in dataset_candidates.items():
                    if len(shape) >= 4:
                        chosen_path = key
                        break
            if chosen_path is None:
                raise RuntimeError(
                    "Could not infer dataset_key from HDF5 datasets. "
                    f"Found keys: {root_keys}"
                )
            chosen_key = chosen_path
            shape = dataset_candidates[chosen_path]

        layout = _infer_layout_from_shape(shape)

        axis = {name: idx for idx, name in enumerate(layout)}
        if "N" in axis:
            n_samples = int(shape[axis["N"]])
        elif sample_grouped:
            n_samples = int(len(sample_groups))
        else:
            n_samples = 0
        n_time = int(shape[axis["T"]]) if "T" in axis else 0
        n_x = int(shape[axis["H"]]) if "H" in axis else 0
        n_y = int(shape[axis["W"]]) if "W" in axis else 0
        n_channels = int(shape[axis["C"]]) if "C" in axis else 1

        dt = None
        t_series = None
        if sample_grouped and sample_group_key is not None:
            for t_key in (f"{sample_group_key}/grid/t", f"{sample_group_key}/t"):
                if t_key in dataset_candidates:
                    t_series = np.asarray(handle[t_key], dtype=np.float64)
                    break
        if t_series is None:
            for t_key in ("t-coordinate", "time", "t"):
                if t_key in handle:
                    t_series = np.asarray(handle[t_key], dtype=np.float64)
                    break
        if t_series is not None and t_series.size >= 2:
            if t_series.ndim > 1:
                t_series = t_series.reshape(-1, t_series.shape[-1])[0]
            diffs = np.diff(t_series)
            finite = diffs[np.isfinite(diffs)]
            if finite.size > 0:
                candidate = float(np.median(np.abs(finite)))
                if candidate > 0.0:
                    dt = candidate

        nu = None
        attr_sources = [handle.attrs]
        data_obj = handle.get(chosen_path)
        if data_obj is not None:
            attr_sources.append(data_obj.attrs)
        if sample_grouped and sample_group_key is not None:
            sample_obj = handle.get(sample_group_key)
            if sample_obj is not None:
                attr_sources.append(sample_obj.attrs)
        for attrs in attr_sources:
            for attr_key in attrs.keys():
                normalized = str(attr_key).strip().lower()
                if any(token in normalized for token in H5_NUMERIC_KINEMATIC_KEYS):
                    value = _float_from_any(attrs[attr_key])
                    if value is not None and value > 0.0:
                        nu = value
                        break
            if nu is not None:
                break

        rd_config_blob: Dict[str, Any] = {}
        rd_physics: Dict[str, float] = {}
        rd_domain: Dict[str, float] = {}
        if sample_grouped and sample_group_key is not None:
            sample_obj = handle.get(sample_group_key)
            if sample_obj is not None:
                rd_config_blob = _extract_rd_yaml_config(sample_obj.attrs.get("config"))
                rd_physics = _extract_rd_physics_from_config_blob(rd_config_blob)
                rd_domain = _extract_rd_domain_from_config_blob(rd_config_blob)
        rd_sim = rd_config_blob.get("sim", {}) if isinstance(rd_config_blob.get("sim", {}), dict) else {}

        if not rd_domain:
            x_series = None
            y_series = None
            if sample_grouped and sample_group_key is not None:
                x_key = f"{sample_group_key}/grid/x"
                y_key = f"{sample_group_key}/grid/y"
                if x_key in dataset_candidates:
                    x_series = np.asarray(handle[x_key], dtype=np.float64)
                if y_key in dataset_candidates:
                    y_series = np.asarray(handle[y_key], dtype=np.float64)
            if x_series is not None and x_series.size >= 2:
                rd_domain["Lx"] = float(np.max(x_series) - np.min(x_series) + np.abs(x_series[1] - x_series[0]))
            if y_series is not None and y_series.size >= 2:
                rd_domain["Ly"] = float(np.max(y_series) - np.min(y_series) + np.abs(y_series[1] - y_series[0]))
        if "t_final" not in rd_domain and t_series is not None and t_series.size >= 2:
            rd_domain["t_final"] = float(np.max(t_series) - np.min(t_series))

    return {
        "dataset_key": chosen_key,
        "dataset_path": chosen_path,
        "shape": shape,
        "layout": layout,
        "sample_grouped": sample_grouped,
        "sample_group_key": sample_group_key,
        "n_samples": n_samples,
        "n_time": n_time,
        "n_x": n_x,
        "n_y": n_y,
        "n_channels": n_channels,
        "dt": dt,
        "nu": nu,
        "rd_physics": rd_physics,
        "rd_domain": rd_domain,
        "rd_sim": rd_sim,
        "root_keys": root_keys,
    }


def _alignment_preview_from_filename(file_path: Path, pde_name: str) -> Dict[str, Any]:
    """Fallback alignment when file is unavailable (e.g., dry-run)."""
    filename = file_path.name.lower()
    if pde_name == "ns_incom" or filename.startswith("ns_incom"):
        return {
            "data.external.pdebench.dataset_key": "velocity",
            "data.external.pdebench.layout": "NTHWC",
            "physics.forcing.type": "none",
            "physics.forcing.amplitude": 0.0,
        }
    if pde_name in RD_PDE_NAMES:
        return {
            "data.external.pdebench.dataset_key": "data",
            "data.external.pdebench.layout": "THWC",
            "data.external.pdebench.sample_grouped": True,
        }
    return {
        "data.external.pdebench.dataset_key": "tensor",
        "data.external.pdebench.layout": "AUTO",
    }


def _safe_int(mapping: Dict[str, Any], key: str, fallback: int) -> int:
    value = mapping.get(key, fallback)
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _resolve_train_test_counts(data_cfg: Dict[str, Any], n_samples: int) -> Tuple[int, int]:
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for train/test split, got {n_samples}.")

    requested_train = _safe_int(data_cfg, "n_train_trajectories", 0)
    requested_test = _safe_int(data_cfg, "n_test_trajectories", 0)

    if requested_train <= 0 and requested_test <= 0:
        n_test = int(round(n_samples * 0.2))
        n_test = max(1, min(n_samples - 1, n_test))
        n_train = n_samples - n_test
        return int(n_train), int(n_test)

    if requested_train <= 0:
        requested_train = max(1, n_samples - max(1, requested_test))
    if requested_test <= 0:
        requested_test = max(1, n_samples - max(1, requested_train))

    total = int(requested_train + requested_test)
    if total <= n_samples and requested_train >= 1 and requested_test >= 1:
        return int(requested_train), int(requested_test)

    test_fraction = float(requested_test) / float(max(total, 1))
    n_test = int(round(n_samples * test_fraction))
    n_test = max(1, min(n_samples - 1, n_test))
    n_train = n_samples - n_test
    return int(n_train), int(n_test)


def align_config_with_downloaded_h5(
    raw_config: Dict[str, Any],
    file_path: Path,
    pde_name: str,
    inspection: Dict[str, Any] | None,
    set_source: bool,
) -> Dict[str, Any]:
    """Align config fields required for robust PDEBench-backed NS runs."""
    report: Dict[str, Any] = {}

    data = raw_config.setdefault("data", {})
    if not isinstance(data, dict):
        raise ValueError("Config key 'data' must be a mapping.")
    external = data.setdefault("external", {})
    if not isinstance(external, dict):
        raise ValueError("Config key 'data.external' must be a mapping.")
    pdebench_cfg = external.setdefault("pdebench", {})
    if not isinstance(pdebench_cfg, dict):
        raise ValueError("Config key 'data.external.pdebench' must be a mapping.")

    pdebench_cfg["file_path"] = str(file_path.resolve())
    report["data.external.pdebench.file_path"] = pdebench_cfg["file_path"]

    if set_source:
        external["source"] = "pdebench"
        report["data.external.source"] = "pdebench"

    if inspection is None:
        preview = _alignment_preview_from_filename(file_path, pde_name)
        pdebench_cfg["dataset_key"] = preview["dataset_key"]
        pdebench_cfg["layout"] = preview["layout"]
        report["data.external.pdebench.dataset_key"] = pdebench_cfg["dataset_key"]
        report["data.external.pdebench.layout"] = pdebench_cfg["layout"]
        return report

    pdebench_cfg["dataset_key"] = str(inspection["dataset_key"])
    pdebench_cfg["layout"] = str(inspection["layout"])
    pdebench_cfg["channel_index"] = 0
    pdebench_cfg["time_stride"] = max(1, int(pdebench_cfg.get("time_stride", 1)))
    report["data.external.pdebench.dataset_key"] = pdebench_cfg["dataset_key"]
    report["data.external.pdebench.layout"] = pdebench_cfg["layout"]
    report["data.external.pdebench.channel_index"] = 0
    report["data.external.pdebench.time_stride"] = pdebench_cfg["time_stride"]
    sample_grouped = bool(inspection.get("sample_grouped", False))
    if sample_grouped:
        pdebench_cfg["sample_grouped"] = True
        report["data.external.pdebench.sample_grouped"] = True
    rd_sim = inspection.get("rd_sim")
    if isinstance(rd_sim, dict) and rd_sim:
        pdebench_cfg["sim"] = dict(rd_sim)
        report["data.external.pdebench.sim"] = pdebench_cfg["sim"]

    dt_value = inspection.get("dt")
    if isinstance(dt_value, (int, float)) and float(dt_value) > 0.0:
        pdebench_cfg["dt"] = float(dt_value)
        report["data.external.pdebench.dt"] = pdebench_cfg["dt"]

    grid = raw_config.setdefault("grid", {})
    if not isinstance(grid, dict):
        raise ValueError("Config key 'grid' must be a mapping.")

    source_nx = int(inspection.get("n_x", 0))
    source_ny = int(inspection.get("n_y", 0))
    if source_nx > 0 and source_ny > 0:
        current_nx = _safe_int(grid, "nx", source_nx)
        current_ny = _safe_int(grid, "ny", source_ny)

        if current_nx > source_nx:
            grid["nx"] = source_nx
            current_nx = source_nx
            report["grid.nx"] = source_nx
        if current_ny > source_ny:
            grid["ny"] = source_ny
            current_ny = source_ny
            report["grid.ny"] = source_ny

        if source_nx % max(current_nx, 1) == 0 and source_ny % max(current_ny, 1) == 0:
            stride_x = source_nx // max(current_nx, 1)
            stride_y = source_ny // max(current_ny, 1)
            if stride_x == stride_y and stride_x >= 1:
                pdebench_cfg["spatial_stride"] = int(stride_x)
                report["data.external.pdebench.spatial_stride"] = int(stride_x)
    pdebench_cfg["spatial_stride"] = max(1, int(pdebench_cfg.get("spatial_stride", 1)))
    report["data.external.pdebench.spatial_stride"] = int(pdebench_cfg["spatial_stride"])

    source_n_samples = int(inspection.get("n_samples", 0))
    if source_n_samples > 1:
        n_train, n_test = _resolve_train_test_counts(data, source_n_samples)
        data["n_train_trajectories"] = int(n_train)
        data["n_test_trajectories"] = int(n_test)
        report["data.n_train_trajectories"] = int(n_train)
        report["data.n_test_trajectories"] = int(n_test)
        pdebench_cfg["n_train"] = int(n_train)
        pdebench_cfg["n_test"] = int(n_test)
        report["data.external.pdebench.n_train"] = int(n_train)
        report["data.external.pdebench.n_test"] = int(n_test)
        pdebench_cfg["shuffle"] = bool(pdebench_cfg.get("shuffle", True))
        pdebench_cfg["split_seed_offset"] = int(pdebench_cfg.get("split_seed_offset", 1207))
        report["data.external.pdebench.shuffle"] = bool(pdebench_cfg["shuffle"])
        report["data.external.pdebench.split_seed_offset"] = int(pdebench_cfg["split_seed_offset"])

    time_cfg = raw_config.setdefault("time", {})
    if not isinstance(time_cfg, dict):
        raise ValueError("Config key 'time' must be a mapping.")
    source_nt = int(inspection.get("n_time", 0))
    if source_nt > 1:
        current_steps = _safe_int(time_cfg, "n_snapshots", source_nt)
        if current_steps > source_nt:
            time_cfg["n_snapshots"] = source_nt
            current_steps = source_nt
            report["time.n_snapshots"] = source_nt
        if isinstance(dt_value, (int, float)) and float(dt_value) > 0.0:
            t_final = float(dt_value) * float(max(current_steps - 1, 1))
            time_cfg["t_final"] = t_final
            report["time.t_final"] = t_final

    physics = raw_config.setdefault("physics", {})
    if not isinstance(physics, dict):
        raise ValueError("Config key 'physics' must be a mapping.")

    if pde_name in RD_PDE_NAMES:
        n_channels = int(inspection.get("n_channels", 1))
        pdebench_cfg["u_channel_index"] = int(pdebench_cfg.get("u_channel_index", 0))
        default_v_idx = 1 if n_channels > 1 else 0
        pdebench_cfg["v_channel_index"] = int(pdebench_cfg.get("v_channel_index", default_v_idx))
        report["data.external.pdebench.u_channel_index"] = int(pdebench_cfg["u_channel_index"])
        report["data.external.pdebench.v_channel_index"] = int(pdebench_cfg["v_channel_index"])

        rd_physics = inspection.get("rd_physics", {})
        if not isinstance(rd_physics, dict):
            rd_physics = {}
        missing_physics = []
        for key in ("Du", "Dv", "F", "k"):
            value = rd_physics.get(key)
            if isinstance(value, (int, float)):
                physics[key] = float(value)
                report[f"physics.{key}"] = float(value)
            else:
                missing_physics.append(key)
        if missing_physics:
            pdebench_cfg["missing_physics"] = missing_physics
            report["data.external.pdebench.missing_physics"] = missing_physics

        rd_domain = inspection.get("rd_domain", {})
        if not isinstance(rd_domain, dict):
            rd_domain = {}
        lx_value = rd_domain.get("Lx")
        ly_value = rd_domain.get("Ly")
        if isinstance(lx_value, (int, float)) and float(lx_value) > 0.0:
            grid["Lx"] = float(lx_value)
            report["grid.Lx"] = float(lx_value)
        if isinstance(ly_value, (int, float)) and float(ly_value) > 0.0:
            grid["Ly"] = float(ly_value)
            report["grid.Ly"] = float(ly_value)

        if source_nt > 1:
            time_cfg["n_snapshots"] = source_nt
            report["time.n_snapshots"] = source_nt
            if isinstance(dt_value, (int, float)) and float(dt_value) > 0.0:
                time_cfg["dt"] = float(dt_value)
                report["time.dt"] = float(dt_value)
                domain_t = rd_domain.get("t_final")
                if isinstance(domain_t, (int, float)) and float(domain_t) > 0.0:
                    time_cfg["t_final"] = float(domain_t)
                else:
                    time_cfg["t_final"] = float(dt_value) * float(max(source_nt - 1, 1))
                report["time.t_final"] = float(time_cfg["t_final"])

    else:
        forcing = physics.setdefault("forcing", {})
        if not isinstance(forcing, dict):
            forcing = {}
            physics["forcing"] = forcing

        # Disable synthetic forcing defaults when switching to external data unless user overrides later.
        forcing["type"] = "none"
        forcing["amplitude"] = 0.0
        report["physics.forcing.type"] = "none"
        report["physics.forcing.amplitude"] = 0.0

        nu_value = inspection.get("nu")
        if isinstance(nu_value, (int, float)) and float(nu_value) > 0.0:
            physics["nu"] = float(nu_value)
            report["physics.nu"] = float(nu_value)

    return report


def patch_config_file(
    config_path: Path,
    file_path: Path,
    pde_name: str,
    set_source: bool,
    align_run_config: bool,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"[dry-run] patch {config_path} data.external.pdebench.file_path={file_path.resolve()}")
        if set_source:
            print(f"[dry-run] patch {config_path} data.external.source=pdebench")
        if align_run_config:
            preview = _alignment_preview_from_filename(file_path, pde_name)
            for key, value in preview.items():
                print(f"[dry-run] patch {config_path} {key}={json.dumps(value)}")
        return

    if not config_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping.")

    data = raw.setdefault("data", {})
    if not isinstance(data, dict):
        raise ValueError("Config key 'data' must be a mapping.")

    external = data.setdefault("external", {})
    if not isinstance(external, dict):
        raise ValueError("Config key 'data.external' must be a mapping.")

    if set_source:
        external["source"] = "pdebench"

    pdebench_cfg = external.setdefault("pdebench", {})
    if not isinstance(pdebench_cfg, dict):
        raise ValueError("Config key 'data.external.pdebench' must be a mapping.")

    inspection = None
    if align_run_config:
        inspection = inspect_h5_for_alignment(file_path)
        print(
            "[info] inspected HDF5 "
            f"dataset_key={inspection['dataset_key']} "
            f"shape={inspection['shape']} layout={inspection['layout']} "
            f"dt={inspection['dt']}"
        )

    report = align_config_with_downloaded_h5(
        raw_config=raw,
        file_path=file_path,
        pde_name=pde_name,
        inspection=inspection,
        set_source=set_source,
    )

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    print(f"[ok] patched {config_path}")
    for key, value in report.items():
        print(f"[ok]   {key} = {json.dumps(value)}")


def main() -> None:
    args = parse_args()

    root_folder = Path(args.root_folder).expanduser()
    config_path = Path(args.config_yaml).expanduser()
    pde_names = [name.strip().lower() for name in args.pde_names]
    filename_contains = args.filename_contains
    max_files = int(args.max_files)
    set_source = bool(args.autofill_source) and not bool(args.no_autofill_source)
    align_run_config = bool(args.align_run_config) and not bool(args.no_align_run_config)

    rows = load_metadata_rows(args.metadata_url)
    selected_rows = filter_rows(rows, pde_names, filename_contains, max_files=max_files)
    if not selected_rows:
        raise RuntimeError(
            "No files matched filters. "
            "Adjust --pde-name / --filename-contains / --max-files."
        )

    print(f"metadata rows matched: {len(selected_rows)}")
    downloaded_paths = [download_row(root_folder, row, dry_run=args.dry_run) for row in selected_rows]

    if args.skip_config_update:
        print("[info] config update skipped")
        return

    chosen_file = downloaded_paths[0]
    chosen_row = selected_rows[0]
    pde_name = str(chosen_row["PDE"]).strip().lower()
    patch_config_file(
        config_path=config_path,
        file_path=chosen_file,
        pde_name=pde_name,
        set_source=set_source,
        align_run_config=align_run_config,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
