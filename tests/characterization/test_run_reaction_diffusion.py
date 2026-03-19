from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data.reaction_diffusion import GrayScottConfig
import runs.run_reaction_diffusion as rd_run


def test_main_reaction_diffusion_uses_run_single_seed_and_writes_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    ckpt_dir = tmp_path / "ckpt"
    out_dir.mkdir()
    ckpt_dir.mkdir()

    args = argparse.Namespace(
        config_yaml=str(tmp_path / "config.yaml"),
        method="tfno",
        seed=7,
        device=None,
        loss=None,
        basis=None,
        resume_clean_checkpoint=None,
        resume_noisy_checkpoint=None,
    )

    raw_config: Dict[str, Any] = {
        "paths": {"output_dir": "output", "checkpoint_dir": "checkpoints"},
        "experiment": {"name": "reaction_diffusion"},
        "training": {
            "device": "cpu",
            "loss": "combined",
            "neural_operator": {},
            "baseline_models": {},
        },
        "rsd": {"basis": "fourier", "spectral_band_count": 8},
        "artifacts": {
            "eval_pair_index": 3,
            "test_case_index": 1,
            "test_step_index": 2,
            "save_figures": False,
            "save_fit_visualizations": False,
            "save_trajectory_visualizations": False,
        },
        "progress": {"enabled": False},
        "data": {"external": {"source": "generated"}},
    }

    run_call: Dict[str, Any] = {}
    save_json_call: Dict[str, Any] = {}

    def _fake_run_single_seed(*run_args: Any, **run_kwargs: Any) -> Dict[str, Any]:
        run_call["args"] = run_args
        run_call["kwargs"] = run_kwargs
        return {
            "clean_l2": 0.1,
            "noisy_l2": 0.2,
            "_viz": {
                "indices": {
                    "eval_pair_index": 3,
                    "test_case_index": 1,
                    "test_step_index": 2,
                }
            },
            "_resolved_device": "cpu",
            "_data": {
                "source": "generated",
                "dt": 0.5,
                "n_snapshots": 4,
                "n_train": 2,
                "n_test": 1,
                "metadata": {},
            },
        }

    def _fake_save_json(path: Path, payload: Dict[str, Any]) -> None:
        save_json_call["path"] = path
        save_json_call["payload"] = payload

    monkeypatch.setattr(rd_run, "parse_args", lambda: args)
    monkeypatch.setattr(rd_run, "load_yaml_config", lambda _: raw_config)
    monkeypatch.setattr(rd_run.GrayScottConfig, "from_yaml", classmethod(lambda cls, _: GrayScottConfig()))
    monkeypatch.setattr(rd_run, "build_run_dirs", lambda *a, **k: (out_dir, ckpt_dir))
    monkeypatch.setattr(rd_run, "run_single_seed", _fake_run_single_seed)
    monkeypatch.setattr(rd_run, "save_json", _fake_save_json)

    rd_run.main()

    assert run_call
    kwargs = run_call["kwargs"]
    assert kwargs["device"] == "cpu"
    assert kwargs["loss"] == "combined"
    assert kwargs["basis"] == "fourier"
    assert kwargs["spectral_band_count"] == 8
    assert kwargs["checkpoint_dir"] == ckpt_dir
    assert kwargs["checkpoint_every_epochs"] == GrayScottConfig().train_checkpoint_every_epochs

    assert save_json_call
    assert save_json_call["path"] == out_dir / "results.json"
    summary = save_json_call["payload"]
    assert summary["experiment"] == "reaction_diffusion"
    assert summary["method"] == "tfno"
    assert summary["seed"] == 7
    assert summary["device_requested"] == "cpu"
    assert summary["device_resolved"] == "cpu"
    assert summary["loss"] == "combined"
    assert summary["basis"] == "fourier"
    assert summary["spectral_band_count"] == 8
    assert summary["metrics"]["clean_l2"] == 0.1
    assert summary["metrics"]["noisy_l2"] == 0.2
    assert summary["viz_indices"] == {"eval_pair_index": 3, "test_case_index": 1, "test_step_index": 2}
    assert summary["data"]["source"] == "generated"
