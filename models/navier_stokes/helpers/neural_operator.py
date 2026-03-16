"""Neural-operator construction helpers for Navier-Stokes surrogates."""

from __future__ import annotations

import io
import inspect
from contextlib import redirect_stdout
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch.nn as nn

try:
    from neuralop.models import FNO, TFNO, UNO

    _NEURALOP_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency path
    FNO = TFNO = UNO = None  # type: ignore[assignment]
    _NEURALOP_IMPORT_ERROR = exc


def require_neuralop() -> None:
    """Ensure neuraloperator is importable before building NO-based models."""
    if FNO is not None and TFNO is not None and UNO is not None:
        return

    message = (
        "Method requires the 'neuraloperator' package. "
        "Install it with: python3 -m pip install neuraloperator"
    )
    if _NEURALOP_IMPORT_ERROR is not None:
        raise ImportError(f"{message}. Import error: {_NEURALOP_IMPORT_ERROR}") from _NEURALOP_IMPORT_ERROR
    raise ImportError(message)


def _normalize_optional_name(value: Any) -> Any:
    """Map textual null markers to Python None."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"none", "null", ""}:
            return None
    return value


def _resolve_modes_2d(value: Any, nx: int, ny: int, default: int = 20) -> Tuple[int, int]:
    """Parse n_modes spec into a valid (mx, my) tuple for 2D operators."""
    max_x = max(2, int(nx) // 2)
    max_y = max(2, int(ny) // 2)

    if isinstance(value, (int, float)):
        mx = my = int(value)
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        mx = int(value[0])
        my = int(value[1])
    else:
        mx = min(max_x, int(default))
        my = min(max_y, int(default))

    mx = int(np.clip(mx, 2, max_x))
    my = int(np.clip(my, 2, max_y))
    return mx, my


def _constructor_accepts_kwargs(model_cls: Any) -> bool:
    """Whether model constructor has a **kwargs sink."""
    try:
        signature = inspect.signature(model_cls.__init__)
    except (TypeError, ValueError):
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def _constructor_param_names(model_cls: Any) -> set[str]:
    """Best-effort set of explicit constructor parameter names."""
    try:
        signature = inspect.signature(model_cls.__init__)
    except (TypeError, ValueError):
        return set()
    return {name for name in signature.parameters if name != "self"}


def _filtered_ctor_kwargs(model_cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop unsupported kwargs for strict constructors."""
    if _constructor_accepts_kwargs(model_cls):
        return kwargs
    supported = _constructor_param_names(model_cls)
    return {key: value for key, value in kwargs.items() if key in supported}


def _add_channel_config_kwargs(
    model_cls: Any,
    kwargs: Dict[str, Any],
    hidden_channels: int,
    lifting_ratio: float,
    projection_ratio: float,
) -> Dict[str, Any]:
    """Populate lifting/projection args in a version-safe way.

    Some neuralop TFNO versions compute channel counts directly from
    `*_channel_ratio` without integer casting, so we prefer explicit
    integer channel counts when available.
    """
    params = _constructor_param_names(model_cls)

    resolved_lifting_channels = max(1, int(round(hidden_channels * max(lifting_ratio, 1e-6))))
    resolved_projection_channels = max(1, int(round(hidden_channels * max(projection_ratio, 1e-6))))

    if "lifting_channels" in params:
        kwargs["lifting_channels"] = resolved_lifting_channels
    elif "lifting_channel_ratio" in params:
        rounded = int(round(lifting_ratio))
        kwargs["lifting_channel_ratio"] = rounded if abs(lifting_ratio - rounded) < 1e-8 else lifting_ratio

    if "projection_channels" in params:
        kwargs["projection_channels"] = resolved_projection_channels
    elif "projection_channel_ratio" in params:
        rounded = int(round(projection_ratio))
        kwargs["projection_channel_ratio"] = rounded if abs(projection_ratio - rounded) < 1e-8 else projection_ratio

    return kwargs


def _add_channel_mlp_kwargs(
    model_cls: Any,
    kwargs: Dict[str, Any],
    use_channel_mlp: bool,
    channel_mlp_dropout: float,
    channel_mlp_expansion: float,
    channel_mlp_skip: str,
) -> Dict[str, Any]:
    """Populate channel-MLP args across neuralop API variants."""
    params = _constructor_param_names(model_cls)

    use_key = "use_channel_mlp" if "use_channel_mlp" in params else "use_mlp"
    dropout_key = "channel_mlp_dropout" if "channel_mlp_dropout" in params else "mlp_dropout"
    expansion_key = "channel_mlp_expansion" if "channel_mlp_expansion" in params else "mlp_expansion"
    skip_key = "channel_mlp_skip" if "channel_mlp_skip" in params else "mlp_skip"

    kwargs[use_key] = bool(use_channel_mlp)
    kwargs[dropout_key] = float(channel_mlp_dropout)
    kwargs[expansion_key] = float(channel_mlp_expansion)
    kwargs[skip_key] = str(channel_mlp_skip)
    return kwargs


def resolve_operator_config(operator: str, operator_config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge common and operator-specific YAML config for neural operators."""
    merged: Dict[str, Any] = {}
    if not isinstance(operator_config, Mapping):
        return merged

    common = operator_config.get("common")
    if isinstance(common, Mapping):
        merged.update(common)

    specific = operator_config.get(operator)
    if isinstance(specific, Mapping):
        merged.update(specific)

    return merged


def build_fno_like_model(
    operator: str,
    in_channels: int,
    out_channels: int,
    nx: int,
    ny: int,
    config: Dict[str, Any],
    n_modes_override: Optional[Sequence[int]] = None,
) -> nn.Module:
    """Construct an FNO/TFNO/UNO model from merged config."""
    require_neuralop()

    modes_x, modes_y = _resolve_modes_2d(config.get("n_modes"), nx=nx, ny=ny, default=20)
    if n_modes_override is None:
        n_modes = (modes_x, modes_y)
    else:
        if len(n_modes_override) < 2:
            raise ValueError("n_modes_override must contain at least two dimensions.")
        n_modes = tuple(max(2, int(mode)) for mode in n_modes_override)

    hidden_channels = max(16, int(config.get("hidden_channels", 64)))
    domain_padding = _normalize_optional_name(config.get("domain_padding", 0.0))
    norm = _normalize_optional_name(config.get("norm"))
    implementation = str(config.get("implementation", "factorized"))
    separable = bool(config.get("separable", False))
    channel_mlp_dropout = float(config.get("channel_mlp_dropout", 0.0))
    channel_mlp_expansion = float(config.get("channel_mlp_expansion", 0.5))
    channel_mlp_skip = str(config.get("channel_mlp_skip", "soft-gating"))
    fno_skip = str(config.get("fno_skip", "linear"))
    use_channel_mlp = bool(config.get("use_channel_mlp", True))
    lifting_ratio = float(config.get("lifting_channel_ratio", 2.0))
    projection_ratio = float(config.get("projection_channel_ratio", 2.0))

    factorization = _normalize_optional_name(config.get("factorization"))
    if isinstance(factorization, str):
        factorization = factorization.strip()

    if operator == "fno":
        fno_kwargs: Dict[str, Any] = dict(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=max(1, int(config.get("n_layers", 6))),
            positional_embedding=str(config.get("positional_embedding", "grid")),
            norm=norm,
            fno_skip=fno_skip,
            domain_padding=domain_padding,
            separable=separable,
            factorization=factorization,
            rank=float(config.get("rank", 1.0)),
            implementation=implementation,
        )
        fno_kwargs = _add_channel_mlp_kwargs(
            FNO,
            fno_kwargs,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
        )
        fno_kwargs = _add_channel_config_kwargs(
            FNO,
            fno_kwargs,
            hidden_channels=hidden_channels,
            lifting_ratio=lifting_ratio,
            projection_ratio=projection_ratio,
        )
        return FNO(**_filtered_ctor_kwargs(FNO, fno_kwargs))

    if operator == "tfno":
        tfno_kwargs: Dict[str, Any] = dict(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=max(1, int(config.get("n_layers", 6))),
            positional_embedding=str(config.get("positional_embedding", "grid")),
            norm=norm,
            fno_skip=fno_skip,
            domain_padding=domain_padding,
            separable=separable,
            factorization=factorization if factorization is not None else "Tucker",
            rank=float(config.get("rank", 0.2)),
            implementation=implementation,
        )
        tfno_kwargs = _add_channel_mlp_kwargs(
            TFNO,
            tfno_kwargs,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
        )
        tfno_kwargs = _add_channel_config_kwargs(
            TFNO,
            tfno_kwargs,
            hidden_channels=hidden_channels,
            lifting_ratio=lifting_ratio,
            projection_ratio=projection_ratio,
        )
        return TFNO(**_filtered_ctor_kwargs(TFNO, tfno_kwargs))

    if operator == "uno":
        if len(n_modes) != 2:
            raise ValueError("UNO currently supports only 2D inputs in this project.")
        n_layers = max(2, int(config.get("n_layers", 5)))
        uno_channel_mlp_skip = str(config.get("channel_mlp_skip", "linear"))
        uno_out_channels = config.get("uno_out_channels")
        if not isinstance(uno_out_channels, list) or len(uno_out_channels) != n_layers:
            uno_out_channels = [hidden_channels for _ in range(n_layers)]
        uno_out_channels = [max(8, int(ch)) for ch in uno_out_channels]

        uno_n_modes = config.get("uno_n_modes")
        if not isinstance(uno_n_modes, list) or len(uno_n_modes) != n_layers:
            uno_n_modes = [[modes_x, modes_y] for _ in range(n_layers)]
        else:
            parsed_modes: List[List[int]] = []
            for layer_modes in uno_n_modes:
                mx, my = _resolve_modes_2d(layer_modes, nx=nx, ny=ny, default=min(modes_x, modes_y))
                parsed_modes.append([mx, my])
            uno_n_modes = parsed_modes

        uno_scalings = config.get("uno_scalings")
        if not isinstance(uno_scalings, list) or len(uno_scalings) != n_layers:
            uno_scalings = [[1.0, 1.0] for _ in range(n_layers)]
        else:
            parsed_scalings: List[List[float]] = []
            for scale in uno_scalings:
                if isinstance(scale, (int, float)):
                    parsed_scalings.append([float(scale), float(scale)])
                elif isinstance(scale, (list, tuple)) and len(scale) >= 2:
                    parsed_scalings.append([float(scale[0]), float(scale[1])])
                else:
                    parsed_scalings.append([1.0, 1.0])
            uno_scalings = parsed_scalings

        horizontal_skips_cfg = config.get("horizontal_skips_map")
        horizontal_skips_map = None
        if isinstance(horizontal_skips_cfg, Mapping):
            horizontal_skips_map = {}
            for key, value in horizontal_skips_cfg.items():
                horizontal_skips_map[int(key)] = int(value)

        uno_kwargs: Dict[str, Any] = dict(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                lifting_channels=max(16, int(config.get("lifting_channels", 192))),
                projection_channels=max(16, int(config.get("projection_channels", 192))),
                positional_embedding=str(config.get("positional_embedding", "grid")),
                n_layers=n_layers,
                uno_out_channels=uno_out_channels,
                uno_n_modes=uno_n_modes,
                uno_scalings=uno_scalings,
                horizontal_skips_map=horizontal_skips_map,
                norm=norm,
                preactivation=bool(config.get("preactivation", False)),
                fno_skip=fno_skip,
                horizontal_skip=str(config.get("horizontal_skip", "linear")),
                separable=separable,
                factorization=factorization,
                rank=float(config.get("rank", 1.0)),
                implementation=implementation,
                domain_padding=domain_padding,
                verbose=bool(config.get("verbose", False)),
        )
        uno_kwargs = _add_channel_mlp_kwargs(
            UNO,
            uno_kwargs,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=uno_channel_mlp_skip,
        )
        with io.StringIO() as suppressed, redirect_stdout(suppressed):
            return UNO(**_filtered_ctor_kwargs(UNO, uno_kwargs))

    raise ValueError(f"Unsupported neural operator type '{operator}'.")
