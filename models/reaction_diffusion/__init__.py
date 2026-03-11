"""Reaction-diffusion surrogate package."""

from models.losses import LOSS_CHOICES, normalize_loss_name

from .helpers.interfaces import CoupledOneStepModel
from .models.factory import build_model, rollout_coupled
from .models.convolutional_surrogate import ConvolutionalSurrogate2DCoupled
from .models.neural_operator_surrogate import NeuralOperatorSurrogate2DCoupled
from .models.physics_surrogate import PhysicsConsistentSurrogate2DCoupled

__all__ = [
    "LOSS_CHOICES",
    "normalize_loss_name",
    "CoupledOneStepModel",
    "ConvolutionalSurrogate2DCoupled",
    "NeuralOperatorSurrogate2DCoupled",
    "PhysicsConsistentSurrogate2DCoupled",
    "build_model",
    "rollout_coupled",
]
