"""Navier-Stokes surrogate package."""

from models.losses import LOSS_CHOICES, normalize_loss_name

from .helpers.interfaces import OneStepModel
from .models.factory import build_model, rollout_2d
from .models.convolutional_surrogate import ConvolutionalSurrogate2D
from .models.neural_operator_surrogate import NeuralOperatorSurrogate2D

__all__ = [
    "LOSS_CHOICES",
    "normalize_loss_name",
    "OneStepModel",
    "ConvolutionalSurrogate2D",
    "NeuralOperatorSurrogate2D",
    "build_model",
    "rollout_2d",
]
