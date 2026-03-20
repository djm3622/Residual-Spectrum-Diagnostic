"""Neural-operator construction helpers for Navier-Stokes surrogates."""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch.nn as nn

# Prefer local neuraloperator clone when present so in-repo WNO work is used.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_NEURALOP_PATH = _PROJECT_ROOT / "external" / "neuraloperator"
if _LOCAL_NEURALOP_PATH.exists():
    _local_path_str = str(_LOCAL_NEURALOP_PATH)
    if _local_path_str in sys.path:
        sys.path.remove(_local_path_str)
    sys.path.insert(0, _local_path_str)

try:
    from models.implicit_tfno import ImplicitTFNO
except Exception:  # pragma: no cover - optional dependency path
    ImplicitTFNO = None  # type: ignore[assignment]

try:
    from neuralop.models import TFNO, UNO

    _NEURALOP_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency path
    TFNO = UNO = None  # type: ignore[assignment]
    _NEURALOP_IMPORT_ERROR = exc
    ImplicitTFNO = None  # type: ignore[assignment]

try:
    from neuralop.models import RNO
except Exception:  # pragma: no cover - optional dependency path
    RNO = None  # type: ignore[assignment]

try:
    from neuralop.models import WNO
except Exception:  # pragma: no cover - optional dependency path
    try:
        from neuralop.models.wno import WNO  # type: ignore[no-redef]
    except Exception:  # pragma: no cover - optional dependency path
        WNO = None  # type: ignore[assignment]


def require_neuralop(operator: str) -> None:
    """Ensure neuraloperator is importable before building NO-based models."""
    normalized = str(operator).strip().lower().replace("-", "_")
    if normalized == "wno":
        if WNO is not None:
            return
        if _NEURALOP_IMPORT_ERROR is not None:
            message = (
                "Method requires the 'neuraloperator' package. "
                "Install it with: python3 -m pip install neuraloperator"
            )
            raise ImportError(f"{message}. Import error: {_NEURALOP_IMPORT_ERROR}") from _NEURALOP_IMPORT_ERROR
        raise ImportError(
            "Method 'wno' requires a neuraloperator build with WNO support "
            "(expected neuralop.models.WNO or neuralop.models.wno.WNO)."
        )

    if normalized == "rno":
        if RNO is not None:
            return
        if _NEURALOP_IMPORT_ERROR is not None:
            message = (
                "Method requires the 'neuraloperator' package. "
                "Install it with: python3 -m pip install neuraloperator"
            )
            raise ImportError(f"{message}. Import error: {_NEURALOP_IMPORT_ERROR}") from _NEURALOP_IMPORT_ERROR
        raise ImportError(
            "Method 'rno' requires a neuraloperator version that exports neuralop.models.RNO."
        )

    if normalized in {"tfno", "itfno"}:
        if TFNO is not None:
            return
    elif normalized == "uno":
        if UNO is not None:
            return
    elif TFNO is not None and UNO is not None:
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


def _normalize_uno_out_channels(value: Any, n_layers: int, hidden_channels: int) -> list[int]:
    n_layers = max(1, int(n_layers))
    default_value = max(1, int(hidden_channels))
    if not isinstance(value, (list, tuple)):
        return [default_value for _ in range(n_layers)]
    values = []
    for item in value:
        try:
            values.append(max(1, int(item)))
        except Exception:
            continue
    if not values:
        values = [default_value]
    if len(values) < n_layers:
        values.extend([values[-1]] * (n_layers - len(values)))
    elif len(values) > n_layers:
        values = values[:n_layers]
    return values


def _normalize_uno_modes(
    value: Any,
    n_layers: int,
    default_modes: Sequence[Any],
    cast: Any,
) -> list[list[Any]]:
    n_layers = max(1, int(n_layers))
    defaults = [cast(item) for item in default_modes]
    if not defaults:
        defaults = [cast(1)]
    n_dim = len(defaults)
    values: list[list[Any]] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                try:
                    row = []
                    for idx in range(n_dim):
                        if idx < len(item):
                            row.append(cast(item[idx]))
                        else:
                            row.append(defaults[idx])
                    values.append(row)
                except Exception:
                    continue
            elif isinstance(item, (int, float)):
                cast_item = cast(item)
                values.append([cast_item for _ in range(n_dim)])
    if not values:
        values = [defaults]
    if len(values) < n_layers:
        values.extend([list(values[-1])] * (n_layers - len(values)))
    elif len(values) > n_layers:
        values = values[:n_layers]
    return values


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


def _instantiate_with_unexpected_kwarg_retry(model_cls: Any, kwargs: Dict[str, Any]) -> Any:
    """Instantiate model while stripping unexpected kwargs reported at runtime."""
    attempt_kwargs = dict(kwargs)
    while True:
        try:
            return model_cls(**attempt_kwargs)
        except TypeError as exc:
            message = str(exc)
            match = re.search(r"unexpected keyword argument '([^']+)'", message)
            if match is None:
                raise
            bad_key = str(match.group(1))
            if bad_key not in attempt_kwargs:
                raise
            attempt_kwargs.pop(bad_key, None)


def _add_channel_config_kwargs(
    model_cls: Any,
    kwargs: Dict[str, Any],
    hidden_channels: int,
    lifting_ratio: float,
    projection_ratio: float,
    lifting_channels: Any = None,
    projection_channels: Any = None,
) -> Dict[str, Any]:
    """Populate lifting/projection args in a version-safe way.

    Some neuralop TFNO versions compute channel counts directly from
    `*_channel_ratio` without integer casting, so we prefer explicit
    integer channel counts when available.
    """
    params = _constructor_param_names(model_cls)

    if lifting_channels is None:
        resolved_lifting_channels = max(1, int(round(hidden_channels * max(lifting_ratio, 1e-6))))
    else:
        resolved_lifting_channels = max(1, int(lifting_channels))

    if projection_channels is None:
        resolved_projection_channels = max(1, int(round(hidden_channels * max(projection_ratio, 1e-6))))
    else:
        resolved_projection_channels = max(1, int(projection_channels))

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

    # Some neuralop releases expose wrapper signatures as (*args, **kwargs),
    # so static introspection cannot tell which naming convention is active.
    opaque_signature = len(params) == 0 or params.issubset({"args", "kwargs"})
    if opaque_signature:
        kwargs["use_channel_mlp"] = bool(use_channel_mlp)
        kwargs["use_mlp"] = bool(use_channel_mlp)
        kwargs["channel_mlp_dropout"] = float(channel_mlp_dropout)
        kwargs["mlp_dropout"] = float(channel_mlp_dropout)
        kwargs["channel_mlp_expansion"] = float(channel_mlp_expansion)
        kwargs["mlp_expansion"] = float(channel_mlp_expansion)
        kwargs["channel_mlp_skip"] = str(channel_mlp_skip)
        kwargs["mlp_skip"] = str(channel_mlp_skip)
        return kwargs

    use_key = "use_channel_mlp" if "use_channel_mlp" in params else "use_mlp"
    dropout_key = "channel_mlp_dropout" if "channel_mlp_dropout" in params else "mlp_dropout"
    expansion_key = "channel_mlp_expansion" if "channel_mlp_expansion" in params else "mlp_expansion"
    skip_key = "channel_mlp_skip" if "channel_mlp_skip" in params else "mlp_skip"

    kwargs[use_key] = bool(use_channel_mlp)
    kwargs[dropout_key] = float(channel_mlp_dropout)
    kwargs[expansion_key] = float(channel_mlp_expansion)
    kwargs[skip_key] = str(channel_mlp_skip)
    return kwargs


def _merge_operator_section(
    merged: Dict[str, Any],
    section: Any,
) -> None:
    """Merge section and optional selected profile override."""
    if not isinstance(section, Mapping):
        return

    for key, value in section.items():
        if key in {"profiles", "profile"}:
            continue
        merged[key] = value

    profile_name = section.get("profile")
    profiles = section.get("profiles")
    if isinstance(profile_name, str) and isinstance(profiles, Mapping):
        selected = profiles.get(profile_name)
        if isinstance(selected, Mapping):
            merged.update(selected)


def resolve_operator_config(operator: str, operator_config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge common and operator-specific YAML config for neural operators."""
    merged: Dict[str, Any] = {}
    if not isinstance(operator_config, Mapping):
        return merged

    common = operator_config.get("common")
    _merge_operator_section(merged, common)

    specific = operator_config.get(operator)
    if not isinstance(specific, Mapping) and operator == "itfno":
        specific = operator_config.get("tfno")
    _merge_operator_section(merged, specific)

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
    """Construct a TFNO/ITFNO/UNO/RNO/WNO model from merged config."""
    require_neuralop(operator)

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
    use_channel_mlp = bool(config.get("use_channel_mlp", False if operator == "wno" else True))
    lifting_ratio = float(config.get("lifting_channel_ratio", 2.0))
    projection_ratio = float(config.get("projection_channel_ratio", 2.0))
    lifting_channels = _normalize_optional_name(config.get("lifting_channels"))
    projection_channels = _normalize_optional_name(config.get("projection_channels"))

    factorization = _normalize_optional_name(config.get("factorization"))
    if isinstance(factorization, str):
        factorization = factorization.strip()

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
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
        )
        return _instantiate_with_unexpected_kwarg_retry(TFNO, _filtered_ctor_kwargs(TFNO, tfno_kwargs))

    if operator == "itfno":
        implicit_cfg = config.get("implicit", {})
        if not isinstance(implicit_cfg, Mapping):
            implicit_cfg = {}

        itfno_kwargs: Dict[str, Any] = dict(
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
            implicit_steps=max(1, int(implicit_cfg.get("steps", config.get("n_layers", 6)))),
            implicit_dt=float(implicit_cfg.get("dt", 1.0)),
            implicit_relaxation=bool(implicit_cfg.get("relaxation", True)),
        )
        itfno_kwargs = _add_channel_mlp_kwargs(
            TFNO,
            itfno_kwargs,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
        )
        itfno_kwargs = _add_channel_config_kwargs(
            TFNO,
            itfno_kwargs,
            hidden_channels=hidden_channels,
            lifting_ratio=lifting_ratio,
            projection_ratio=projection_ratio,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
        )
        if ImplicitTFNO is None:
            raise ImportError("Method 'itfno' requires neuraloperator TFNO support.")
        return _instantiate_with_unexpected_kwarg_retry(
            ImplicitTFNO,
            _filtered_ctor_kwargs(ImplicitTFNO, itfno_kwargs),
        )

    if operator == "uno":
        uno_layers_cfg = max(1, int(config.get("n_layers", 6)))
        inferred_uno_layers = 0
        for raw in (config.get("uno_out_channels"), config.get("uno_n_modes"), config.get("uno_scalings")):
            if isinstance(raw, (list, tuple)) and len(raw) > 0:
                inferred_uno_layers = max(inferred_uno_layers, len(raw))
        uno_n_layers = max(1, inferred_uno_layers if inferred_uno_layers > 0 else uno_layers_cfg)
        uno_out_channels = _normalize_uno_out_channels(
            config.get("uno_out_channels"),
            n_layers=uno_n_layers,
            hidden_channels=hidden_channels,
        )
        uno_n_modes = _normalize_uno_modes(
            config.get("uno_n_modes"),
            n_layers=uno_n_layers,
            default_modes=[float(mode) for mode in n_modes],
            cast=int,
        )
        uno_scalings = _normalize_uno_modes(
            config.get("uno_scalings"),
            n_layers=uno_n_layers,
            default_modes=[1.0 for _ in n_modes],
            cast=float,
        )
        uno_kwargs: Dict[str, Any] = dict(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=uno_n_layers,
            positional_embedding=str(config.get("positional_embedding", "grid")),
            norm=norm,
            fno_skip=fno_skip,
            domain_padding=domain_padding,
            separable=separable,
            factorization=factorization if factorization is not None else "Tucker",
            rank=float(config.get("rank", 0.2)),
            implementation=implementation,
            uno_out_channels=uno_out_channels,
            uno_n_modes=uno_n_modes,
            uno_scalings=uno_scalings,
        )
        uno_kwargs = _add_channel_mlp_kwargs(
            UNO,
            uno_kwargs,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
        )
        uno_kwargs = _add_channel_config_kwargs(
            UNO,
            uno_kwargs,
            hidden_channels=hidden_channels,
            lifting_ratio=lifting_ratio,
            projection_ratio=projection_ratio,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
        )
        return _instantiate_with_unexpected_kwarg_retry(UNO, _filtered_ctor_kwargs(UNO, uno_kwargs))

    if operator == "rno":
        rno_kwargs: Dict[str, Any] = dict(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=max(1, int(config.get("n_layers", 4))),
            positional_embedding=str(config.get("positional_embedding", "grid")),
            norm=norm,
            fno_skip=fno_skip,
            domain_padding=domain_padding,
            separable=separable,
            factorization=factorization if factorization is not None else "Tucker",
            rank=float(config.get("rank", 0.2)),
            implementation=implementation,
            rno_skip=bool(config.get("rno_skip", False)),
            return_sequences=bool(config.get("return_sequences", False)),
        )
        rno_kwargs = _add_channel_mlp_kwargs(
            RNO,
            rno_kwargs,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
        )
        rno_kwargs = _add_channel_config_kwargs(
            RNO,
            rno_kwargs,
            hidden_channels=hidden_channels,
            lifting_ratio=lifting_ratio,
            projection_ratio=projection_ratio,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
        )
        return _instantiate_with_unexpected_kwarg_retry(RNO, _filtered_ctor_kwargs(RNO, rno_kwargs))

    if operator == "wno":
        raw_base_resolution = _normalize_optional_name(config.get("base_resolution"))
        if isinstance(raw_base_resolution, (list, tuple)):
            base_resolution = [int(v) for v in raw_base_resolution]
        else:
            base_resolution = None

        wno_kwargs: Dict[str, Any] = dict(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=max(1, int(config.get("n_layers", 4))),
            blocks=_normalize_optional_name(config.get("blocks")),
            positional_embedding=str(config.get("positional_embedding", "grid")),
            norm=norm,
            fno_skip=fno_skip,
            domain_padding=domain_padding,
            separable=separable,
            factorization=factorization,
            rank=float(config.get("rank", 1.0)),
            implementation=implementation,
            wavelet=str(config.get("wavelet", "db4")),
            wavelet_levels=max(1, int(config.get("wavelet_levels", 2))),
            wavelet_mode=str(config.get("wavelet_mode", "symmetric")),
            wavelet_transform=str(config.get("wavelet_transform", "dwt")),
            base_resolution=base_resolution,
        )
        wno_kwargs = _add_channel_mlp_kwargs(
            WNO,
            wno_kwargs,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
        )
        wno_kwargs = _add_channel_config_kwargs(
            WNO,
            wno_kwargs,
            hidden_channels=hidden_channels,
            lifting_ratio=lifting_ratio,
            projection_ratio=projection_ratio,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
        )
        return _instantiate_with_unexpected_kwarg_retry(WNO, _filtered_ctor_kwargs(WNO, wno_kwargs))

    raise ValueError(f"Unsupported neural operator type '{operator}'.")
