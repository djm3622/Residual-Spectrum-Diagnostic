from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data.navier_stokes import NSConfig
import runs.run_navier_stokes as ns_run


def test_main_navier_stokes_uses_run_single_seed_and_writes_summary(
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
        seed=13,
        device=None,
        loss=None,
        basis=None,
        resume_clean_checkpoint=None,
        resume_noisy_checkpoint=None,
    )

    raw_config: Dict[str, Any] = {
        "paths": {"output_dir": "output", "checkpoint_dir": "checkpoints"},
        "experiment": {"name": "navier_stokes"},
        "training": {
            "device": "cpu",
            "loss": "combined",
            "neural_operator": {},
            "baseline_models": {},
        },
        "rsd": {"basis": "fourier", "spectral_band_count": 8},
        "artifacts": {
            "eval_pair_index": 4,
            "test_case_index": 2,
            "test_step_index": 1,
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
            "clean_l2": 0.11,
            "noisy_l2": 0.22,
            "_viz": {
                "indices": {
                    "eval_pair_index": 4,
                    "test_case_index": 2,
                    "test_step_index": 1,
                }
            },
            "_resolved_device": "cpu",
            "_data_source": "generated",
            "_data_metadata": {"source": "synthetic"},
            "_dt": 0.25,
            "_n_snapshots": 20,
        }

    def _fake_save_json(path: Path, payload: Dict[str, Any]) -> None:
        save_json_call["path"] = path
        save_json_call["payload"] = payload

    monkeypatch.setattr(ns_run, "parse_args", lambda: args)
    monkeypatch.setattr(ns_run, "load_yaml_config", lambda _: raw_config)
    monkeypatch.setattr(ns_run.NSConfig, "from_yaml", classmethod(lambda cls, _: NSConfig()))
    monkeypatch.setattr(ns_run, "external_data_config_from_yaml", lambda _: object())
    monkeypatch.setattr(ns_run, "build_run_dirs", lambda *a, **k: (out_dir, ckpt_dir))
    monkeypatch.setattr(ns_run, "run_single_seed", _fake_run_single_seed)
    monkeypatch.setattr(ns_run, "save_json", _fake_save_json)

    ns_run.main()

    assert run_call
    kwargs = run_call["kwargs"]
    assert kwargs["device"] == "cpu"
    assert kwargs["loss"] == "combined"
    assert kwargs["basis"] == "fourier"
    assert kwargs["spectral_band_count"] == 8
    assert kwargs["checkpoint_dir"] == ckpt_dir
    assert kwargs["checkpoint_every_epochs"] == NSConfig().train_checkpoint_every_epochs

    assert save_json_call
    assert save_json_call["path"] == out_dir / "results.json"
    summary = save_json_call["payload"]
    assert summary["experiment"] == "navier_stokes"
    assert summary["method"] == "tfno"
    assert summary["seed"] == 13
    assert summary["device_requested"] == "cpu"
    assert summary["device_resolved"] == "cpu"
    assert summary["loss"] == "combined"
    assert summary["basis"] == "fourier"
    assert summary["spectral_band_count"] == 8
    assert summary["metrics"]["clean_l2"] == 0.11
    assert summary["metrics"]["noisy_l2"] == 0.22
    assert summary["data_source"] == "generated"
    assert summary["data_metadata"] == {"source": "synthetic"}
    assert summary["dt"] == 0.25
    assert summary["n_snapshots"] == 20
    assert summary["viz_indices"] == {"eval_pair_index": 4, "test_case_index": 2, "test_step_index": 1}
