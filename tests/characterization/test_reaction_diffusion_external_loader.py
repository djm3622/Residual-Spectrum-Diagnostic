from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from data.reaction_diffusion.external import (
    PDEBenchReactionDiffusionSourceConfig,
    load_pdebench_reaction_diffusion_data,
)
from data.reaction_diffusion.solver import GrayScottConfig


def _base_config(nx: int = 4, ny: int = 4, n_snapshots: int = 4) -> GrayScottConfig:
    return GrayScottConfig(
        nx=nx,
        ny=ny,
        n_snapshots=n_snapshots,
        n_train_trajectories=2,
        n_test_trajectories=1,
        t_final=1.0,
    )


def test_load_pdebench_reaction_diffusion_data_root_dataset(tmp_path: Path) -> None:
    h5_path = tmp_path / "rd_root.h5"
    n_samples, n_steps, nx, ny = 3, 5, 4, 4

    data = np.zeros((n_samples, n_steps, nx, ny, 2), dtype=np.float32)
    for sample_idx in range(n_samples):
        for step_idx in range(n_steps):
            base = sample_idx * 100.0 + step_idx
            data[sample_idx, step_idx, :, :, 0] = base
            data[sample_idx, step_idx, :, :, 1] = base + 0.5

    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("data", data=data)
        handle.create_dataset("t-coordinate", data=np.asarray([0.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float64))

    config = _base_config(nx=nx, ny=ny, n_snapshots=4)
    source_cfg = PDEBenchReactionDiffusionSourceConfig(
        file_path=str(h5_path),
        dataset_key="data",
        layout="AUTO",
        sample_grouped=False,
        n_train=2,
        n_test=1,
        shuffle=False,
        time_stride=1,
        spatial_stride=1,
    )

    loaded = load_pdebench_reaction_diffusion_data(config=config, source_cfg=source_cfg, seed=11)

    assert loaded.source == "pdebench"
    assert loaded.n_snapshots == 4
    assert loaded.dt == 0.2
    assert len(loaded.train_data) == 2
    assert len(loaded.test_cases) == 1

    np.testing.assert_allclose(loaded.train_data[0]["u"], data[0, :4, :, :, 0])
    np.testing.assert_allclose(loaded.train_data[0]["v"], data[0, :4, :, :, 1])
    np.testing.assert_allclose(loaded.train_data[1]["u"], data[1, :4, :, :, 0])
    np.testing.assert_allclose(loaded.train_data[1]["v"], data[1, :4, :, :, 1])

    np.testing.assert_allclose(loaded.test_cases[0]["u_true"], data[2, :4, :, :, 0])
    np.testing.assert_allclose(loaded.test_cases[0]["v_true"], data[2, :4, :, :, 1])
    np.testing.assert_allclose(loaded.test_cases[0]["u0"], data[2, 0, :, :, 0])
    np.testing.assert_allclose(loaded.test_cases[0]["v0"], data[2, 0, :, :, 1])

    assert loaded.metadata["layout"] == "NTHWC"
    assert loaded.metadata["raw_shape"] == (n_samples, n_steps, nx, ny, 2)
    assert loaded.metadata["n_train_loaded"] == 2
    assert loaded.metadata["n_test_loaded"] == 1
    assert loaded.metadata["sample_grouped"] is False


def test_load_pdebench_reaction_diffusion_data_grouped_samples(tmp_path: Path) -> None:
    h5_path = tmp_path / "rd_grouped.h5"
    n_samples, n_steps, src_nx, src_ny = 4, 6, 8, 8

    grouped_data: list[np.ndarray] = []
    with h5py.File(h5_path, "w") as handle:
        for sample_idx in range(n_samples):
            grp = handle.create_group(f"sample_{sample_idx:03d}")
            sample = np.zeros((n_steps, src_nx, src_ny, 2), dtype=np.float32)
            for step_idx in range(n_steps):
                base = sample_idx * 1000.0 + step_idx * 10.0
                sample[step_idx, :, :, 0] = base
                sample[step_idx, :, :, 1] = base + 2.0
            grp.create_dataset("data", data=sample)
            grid = grp.create_group("grid")
            grid.create_dataset("t", data=np.asarray([0.0, 0.05, 0.10, 0.15, 0.20, 0.25], dtype=np.float64))
            grouped_data.append(sample)

    config = _base_config(nx=4, ny=4, n_snapshots=5)
    source_cfg = PDEBenchReactionDiffusionSourceConfig(
        file_path=str(h5_path),
        dataset_key="data",
        layout="AUTO",
        sample_grouped=True,
        n_train=2,
        n_test=1,
        shuffle=False,
        time_stride=2,
        spatial_stride=2,
    )

    loaded = load_pdebench_reaction_diffusion_data(config=config, source_cfg=source_cfg, seed=19)

    assert loaded.source == "pdebench"
    assert loaded.n_snapshots == 3
    assert loaded.dt == 0.1
    assert len(loaded.train_data) == 2
    assert len(loaded.test_cases) == 1

    expected_train0_u = grouped_data[0][::2, ::2, ::2, 0][:3]
    expected_train0_v = grouped_data[0][::2, ::2, ::2, 1][:3]
    expected_train1_u = grouped_data[1][::2, ::2, ::2, 0][:3]
    expected_train1_v = grouped_data[1][::2, ::2, ::2, 1][:3]
    expected_test_u = grouped_data[2][::2, ::2, ::2, 0][:3]
    expected_test_v = grouped_data[2][::2, ::2, ::2, 1][:3]

    np.testing.assert_allclose(loaded.train_data[0]["u"], expected_train0_u)
    np.testing.assert_allclose(loaded.train_data[0]["v"], expected_train0_v)
    np.testing.assert_allclose(loaded.train_data[1]["u"], expected_train1_u)
    np.testing.assert_allclose(loaded.train_data[1]["v"], expected_train1_v)
    np.testing.assert_allclose(loaded.test_cases[0]["u_true"], expected_test_u)
    np.testing.assert_allclose(loaded.test_cases[0]["v_true"], expected_test_v)

    assert loaded.metadata["layout"] == "THWC"
    assert loaded.metadata["raw_shape"] == (n_steps, src_nx, src_ny, 2)
    assert loaded.metadata["n_train_loaded"] == 2
    assert loaded.metadata["n_test_loaded"] == 1
    assert loaded.metadata["sample_grouped"] is True
