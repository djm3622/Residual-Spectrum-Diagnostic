"""Progress bar helpers with graceful fallback when tqdm is unavailable."""

from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None


def progress_iter(
    iterable: Iterable[T],
    enabled: bool = False,
    desc: str | None = None,
    total: int | None = None,
    leave: bool = False,
) -> Iterable[T]:
    """Wrap iterable with tqdm when enabled and installed."""
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True)


def progress_range(
    stop: int,
    enabled: bool = False,
    desc: str | None = None,
    leave: bool = False,
) -> Iterable[int]:
    """range(stop) with optional tqdm progress."""
    return progress_iter(range(stop), enabled=enabled, desc=desc, total=stop, leave=leave)
