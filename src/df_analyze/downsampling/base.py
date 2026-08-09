from __future__ import annotations

from math import ceil, floor
from numbers import Integral, Real
from typing import Union


def resolve_n_features(requested: Union[int, float], n_features: int) -> int:
    if n_features <= 0:
        return 0
    if isinstance(requested, Integral) and not isinstance(requested, bool):
        if requested <= 0:
            raise ValueError("Feature downsample count must be positive.")
        return min(int(requested), n_features)
    if isinstance(requested, Real):
        requested = float(requested)
        if 0.0 < requested <= 1.0:
            return max(1, min(n_features, ceil(requested * n_features)))
        if requested > 1.0 and requested.is_integer():
            return min(int(requested), n_features)
    raise ValueError(
        "Feature downsample size must be a positive integer or a fraction in (0, 1]."
    )


def auto_chunk_size(
    n_rows: int,
    itemsize: int,
    max_chunk_bytes: int,
    requested: int,
    overhead_factor: float = 3.0,
) -> int:
    requested = max(1, int(requested))
    denominator = max(1, n_rows) * max(1, itemsize) * max(1.0, overhead_factor)
    allowed = max(1, int(floor(max_chunk_bytes / denominator)))
    return min(requested, allowed)
