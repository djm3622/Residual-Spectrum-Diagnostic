"""PDEBench split-loading helpers for reaction-diffusion trajectories."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from ..solver import GrayScottConfig
from .external_io import (
    _infer_dt_from_group,
    _infer_dt_from_root,
    _load_grouped_samples,
    _load_root_samples,
    _resolve_layout,
    _sample_group_keys,
)


def load_grouped_pdebench_split(
    handle: Any,
    config: GrayScottConfig,
    source_cfg: Any,
    requested_train: int,
    requested_test: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], float | None, Tuple[int, ...], str]:
    """Load grouped PDEBench samples and split into train/test trajectories."""
    import h5py

    sample_keys = _sample_group_keys(handle)
    available = len(sample_keys)
    if available < requested_train + requested_test:
        raise ValueError(
            f"PDEBench file has {available} grouped samples but "
            f"{requested_train + requested_test} are required."
        )

    indices = np.arange(available, dtype=int)
    if source_cfg.shuffle:
        rng = np.random.default_rng(seed * 1000 + int(source_cfg.split_seed_offset))
        rng.shuffle(indices)
    train_indices = indices[:requested_train]
    test_indices = indices[requested_train : requested_train + requested_test]

    layout_hint = str(source_cfg.layout)
    train_u, train_v = _load_grouped_samples(
        handle=handle,
        sample_keys=sample_keys,
        selected_indices=train_indices.tolist(),
        dataset_key=str(source_cfg.dataset_key),
        layout_hint=layout_hint,
        u_channel_index=int(source_cfg.u_channel_index),
        v_channel_index=int(source_cfg.v_channel_index),
        time_stride=int(source_cfg.time_stride),
        spatial_stride=int(source_cfg.spatial_stride),
        target_nx=int(config.nx),
        target_ny=int(config.ny),
    )
    test_u, test_v = _load_grouped_samples(
        handle=handle,
        sample_keys=sample_keys,
        selected_indices=test_indices.tolist(),
        dataset_key=str(source_cfg.dataset_key),
        layout_hint=layout_hint,
        u_channel_index=int(source_cfg.u_channel_index),
        v_channel_index=int(source_cfg.v_channel_index),
        time_stride=int(source_cfg.time_stride),
        spatial_stride=int(source_cfg.spatial_stride),
        target_nx=int(config.nx),
        target_ny=int(config.ny),
    )
    dt_from_file = _infer_dt_from_group(handle, sample_keys[0])
    first_group = handle[sample_keys[0]]
    first_dataset_key = str(source_cfg.dataset_key).strip() or "data"
    if first_dataset_key not in first_group:
        if "data" in first_group:
            first_dataset_key = "data"
        else:
            first_dataset_key = next(
                (
                    key
                    for key in first_group.keys()
                    if isinstance(first_group.get(key), h5py.Dataset)
                ),
                "",
            )
    if not first_dataset_key:
        raise ValueError(f"Could not infer dataset key from sample group '{sample_keys[0]}'.")
    raw_shape = tuple(int(dim) for dim in first_group[first_dataset_key].shape)
    resolved_layout = _resolve_layout(len(raw_shape), source_cfg.layout, grouped_hint=True)
    return train_u, train_v, test_u, test_v, dt_from_file, raw_shape, resolved_layout


def load_root_pdebench_split(
    handle: Any,
    config: GrayScottConfig,
    source_cfg: Any,
    requested_train: int,
    requested_test: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], float | None, Tuple[int, ...], str]:
    """Load root-dataset PDEBench samples and split into train/test trajectories."""
    root_keys = list(handle.keys())
    dataset_key = str(source_cfg.dataset_key).strip() or "data"
    if dataset_key not in handle:
        for candidate in ("data", "tensor", "fields"):
            if candidate in handle:
                dataset_key = candidate
                break
        else:
            raise KeyError(
                f"Dataset key '{source_cfg.dataset_key}' not found in {handle.filename}. "
                f"Available keys: {root_keys}"
            )

    dataset = handle[dataset_key]
    raw_shape = tuple(int(dim) for dim in dataset.shape)
    resolved_layout = _resolve_layout(dataset.ndim, source_cfg.layout, grouped_hint=False)
    axis_map = {label: idx for idx, label in enumerate(resolved_layout)}
    if "N" not in axis_map:
        raise ValueError(
            f"Root dataset layout must include N for non-grouped data. Got '{resolved_layout}'."
        )
    available = int(dataset.shape[axis_map["N"]])
    if available < requested_train + requested_test:
        raise ValueError(
            f"PDEBench dataset has {available} samples but "
            f"{requested_train + requested_test} are required."
        )

    indices = np.arange(available, dtype=int)
    if source_cfg.shuffle:
        rng = np.random.default_rng(seed * 1000 + int(source_cfg.split_seed_offset))
        rng.shuffle(indices)
    train_indices = indices[:requested_train]
    test_indices = indices[requested_train : requested_train + requested_test]

    train_u, train_v = _load_root_samples(
        dataset=dataset,
        selected_indices=train_indices.tolist(),
        layout=resolved_layout,
        u_channel_index=int(source_cfg.u_channel_index),
        v_channel_index=int(source_cfg.v_channel_index),
        time_stride=int(source_cfg.time_stride),
        spatial_stride=int(source_cfg.spatial_stride),
        target_nx=int(config.nx),
        target_ny=int(config.ny),
    )
    test_u, test_v = _load_root_samples(
        dataset=dataset,
        selected_indices=test_indices.tolist(),
        layout=resolved_layout,
        u_channel_index=int(source_cfg.u_channel_index),
        v_channel_index=int(source_cfg.v_channel_index),
        time_stride=int(source_cfg.time_stride),
        spatial_stride=int(source_cfg.spatial_stride),
        target_nx=int(config.nx),
        target_ny=int(config.ny),
    )
    dt_from_file = _infer_dt_from_root(handle)
    return train_u, train_v, test_u, test_v, dt_from_file, raw_shape, resolved_layout
