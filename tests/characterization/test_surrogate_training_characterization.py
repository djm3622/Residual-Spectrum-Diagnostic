from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import TensorDataset

import models.navier_stokes.models.convolutional_surrogate as ns_conv_mod
import models.navier_stokes.models.neural_operator_surrogate as ns_no_mod
import models.reaction_diffusion.models.convolutional_surrogate as rd_conv_mod
import models.reaction_diffusion.models.neural_operator_surrogate as rd_no_mod


class TinyPointwise(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            # [B, C, T, H, W] -> apply per-frame and restore shape
            b, c, t, h, w = x.shape
            y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            y = self.conv(y)
            y = y.reshape(b, t, y.shape[1], h, w).permute(0, 2, 1, 3, 4)
            return y
        return self.conv(x)


def _fake_build_fno_like_model(
    operator: str,
    in_channels: int,
    out_channels: int,
    nx: int,
    ny: int,
    config: Dict[str, Any],
    n_modes_override: Any = None,
) -> TinyPointwise:
    del operator, nx, ny, config, n_modes_override
    return TinyPointwise(in_channels, out_channels)


def _checkpoint_keys_common() -> set[str]:
    return {
        "epoch",
        "val_loss",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "grad_scaler_state",
        "rng_state",
        "best_val_loss",
        "epochs_without_improvement",
        "best_model_state",
        "loss_name",
    }


def test_reaction_diffusion_convolutional_train_checkpoint_state(monkeypatch) -> None:
    monkeypatch.setattr(rd_conv_mod, "RDUNetOneStepNet", lambda width: TinyPointwise(2, 2))

    model = rd_conv_mod.ConvolutionalSurrogate2DCoupled(nx=4, ny=4, device="cpu", architecture="legacy_conv")
    dataset = TensorDataset(torch.randn(6, 2, 4, 4), torch.randn(6, 2, 4, 4))

    checkpoints: List[Dict[str, Any]] = []

    def _cb(epoch: int, val_loss: float, state: Dict[str, Any]) -> None:
        del val_loss
        assert isinstance(epoch, int)
        checkpoints.append(state)

    model.train(
        inputs_u=[],
        inputs_v=[],
        targets_u=[],
        targets_v=[],
        n_iter=2,
        batch_size=2,
        train_dataset=dataset,
        val_dataset=dataset,
        dataloader_num_workers=0,
        checkpoint_callback=_cb,
        show_progress=False,
    )

    assert checkpoints
    state = checkpoints[-1]
    assert _checkpoint_keys_common().issubset(state.keys())
    assert isinstance(state["model_state"], dict)
    assert isinstance(state["optimizer_state"], dict)
    assert isinstance(state["rng_state"], dict)


def test_reaction_diffusion_neural_operator_train_checkpoint_and_resume(monkeypatch) -> None:
    monkeypatch.setattr(rd_no_mod, "resolve_operator_config", lambda operator, operator_config: {})
    monkeypatch.setattr(rd_no_mod, "build_fno_like_model", _fake_build_fno_like_model)

    model = rd_no_mod.NeuralOperatorSurrogate2DCoupled(nx=4, ny=4, operator="tfno", device="cpu")
    dataset = TensorDataset(torch.randn(6, 2, 4, 4), torch.randn(6, 2, 4, 4))

    checkpoints: List[Dict[str, Any]] = []

    model.train(
        inputs_u=[],
        inputs_v=[],
        targets_u=[],
        targets_v=[],
        n_iter=1,
        batch_size=2,
        train_dataset=dataset,
        val_dataset=dataset,
        dataloader_num_workers=0,
        checkpoint_callback=lambda epoch, val_loss, state: checkpoints.append(state),
        show_progress=False,
    )

    assert checkpoints
    state = checkpoints[-1]
    assert _checkpoint_keys_common().issubset(state.keys())
    assert "temporal_enabled" in state

    resumed = rd_no_mod.NeuralOperatorSurrogate2DCoupled(nx=4, ny=4, operator="tfno", device="cpu")
    resumed_epochs: List[int] = []

    resumed.train(
        inputs_u=[],
        inputs_v=[],
        targets_u=[],
        targets_v=[],
        n_iter=int(state["epoch"]) + 1,
        batch_size=2,
        train_dataset=dataset,
        val_dataset=dataset,
        dataloader_num_workers=0,
        checkpoint_callback=lambda epoch, val_loss, training_state: resumed_epochs.append(epoch),
        resume_state=state,
        show_progress=False,
    )

    assert resumed_epochs
    assert resumed_epochs[0] == int(state["epoch"]) + 1


def test_navier_stokes_convolutional_train_checkpoint_state(monkeypatch) -> None:
    monkeypatch.setattr(ns_conv_mod, "NSNonlinearOneStepNet", lambda width, depth: TinyPointwise(1, 1))

    model = ns_conv_mod.ConvolutionalSurrogate2D(nx=4, ny=4, device="cpu", architecture="legacy_conv")
    dataset = TensorDataset(torch.randn(6, 1, 4, 4), torch.randn(6, 1, 4, 4))

    checkpoints: List[Dict[str, Any]] = []

    model.train(
        inputs=[],
        targets=[],
        n_iter=2,
        batch_size=2,
        train_dataset=dataset,
        val_dataset=dataset,
        dataloader_num_workers=0,
        checkpoint_callback=lambda epoch, val_loss, state: checkpoints.append(state),
        show_progress=False,
    )

    assert checkpoints
    state = checkpoints[-1]
    assert _checkpoint_keys_common().issubset(state.keys())
    assert isinstance(state["model_state"], dict)
    assert isinstance(state["optimizer_state"], dict)


def test_navier_stokes_neural_operator_train_checkpoint_state(monkeypatch) -> None:
    monkeypatch.setattr(ns_no_mod, "resolve_operator_config", lambda operator, operator_config: {})
    monkeypatch.setattr(ns_no_mod, "build_fno_like_model", _fake_build_fno_like_model)

    model = ns_no_mod.NeuralOperatorSurrogate2D(nx=4, ny=4, operator="tfno", device="cpu")
    dataset = TensorDataset(torch.randn(6, 1, 4, 4), torch.randn(6, 1, 4, 4))

    checkpoints: List[Dict[str, Any]] = []

    model.train(
        inputs=[],
        targets=[],
        n_iter=2,
        batch_size=2,
        train_dataset=dataset,
        val_dataset=dataset,
        dataloader_num_workers=0,
        checkpoint_callback=lambda epoch, val_loss, state: checkpoints.append(state),
        show_progress=False,
    )

    assert checkpoints
    state = checkpoints[-1]
    assert _checkpoint_keys_common().issubset(state.keys())
    assert "temporal_enabled" in state
