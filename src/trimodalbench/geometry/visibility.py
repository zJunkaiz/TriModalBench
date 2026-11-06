"""Visibility and depth residuals on the source pixel grid."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .reprojection import Projection


class Visibility(IntEnum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    OCCLUDED = 2
    VISIBLE = 3


@dataclass(frozen=True)
class DepthReprojection:
    state: np.ndarray
    absolute: np.ndarray
    relative: np.ndarray
    selected: np.ndarray
    occlusion_counts: dict[str, int]


def evaluate_depth(
    projection: Projection,
    target_depth: np.ndarray,
    *,
    selected: np.ndarray | None = None,
    external_visible: np.ndarray | None = None,
    reference_depth: np.ndarray | None = None,
    tolerance_m: float = 0.01,
    relative_tolerance: float = 0.001,
) -> DepthReprojection:
    """Score target depth using nearest-pixel Z-buffering and optional visibility evidence.

    Use an independent reference for scoring model output with reference-depth
    occlusion. A scored target reused as reference produces a diagnostic subset.
    """
    shape = projection.depth.shape
    if target_depth.ndim != 2 or not target_depth.size:
        raise ValueError("Expected a nonempty target depth image")
    if reference_depth is not None and reference_depth.shape != target_depth.shape:
        raise ValueError("Reference depth must use the target pixel grid")
    for mask in (selected, external_visible):
        if mask is not None and (mask.shape != shape or mask.dtype != np.bool_):
            raise ValueError("Masks must be boolean arrays on the source grid")
    if not all(np.isfinite(value) and value >= 0 for value in (tolerance_m, relative_tolerance)):
        raise ValueError("Visibility tolerances must be finite and nonnegative")
    if selected is None:
        selected = np.ones(shape, dtype=bool)

    state = np.full(shape, Visibility.INVALID, dtype=np.uint8)
    projectable = selected & projection.positive_depth & np.isfinite(projection.pixels).all(axis=-1)
    state[projectable & ~projection.in_bounds] = Visibility.OUT_OF_BOUNDS
    inside = projectable & projection.in_bounds
    height, width = target_depth.shape
    u, v = np.floor(projection.pixels[inside] + 0.5).astype(np.intp).T
    u, v = np.clip(u, 0, width - 1), np.clip(v, 0, height - 1)
    z = projection.depth[inside]

    z_buffer = np.full((height, width), np.inf)
    np.minimum.at(z_buffer, (v, u), z)
    nearest = z_buffer[v, u]
    behind_source = z - nearest > tolerance_m + relative_tolerance * nearest
    external = np.zeros(z.shape, dtype=bool) if external_visible is None else ~external_visible[inside]
    behind_source &= ~external
    behind_reference = np.zeros(z.shape, dtype=bool)
    if reference_depth is not None:
        reference = reference_depth[v, u]
        known = np.isfinite(reference) & (reference > 0)
        behind_reference[known] = (
            z[known] - reference[known] > tolerance_m + relative_tolerance * reference[known]
        )
        behind_reference &= ~(external | behind_source)

    observed = target_depth[v, u]
    occluded = external | behind_source | behind_reference
    visible = ~occluded & np.isfinite(observed) & (observed > 0)
    states = np.full(z.shape, Visibility.INVALID, dtype=np.uint8)
    states[occluded] = Visibility.OCCLUDED
    states[visible] = Visibility.VISIBLE
    state[inside] = states
    absolute = np.full(shape, np.nan)
    relative = np.full(shape, np.nan)
    error = np.abs(z[visible] - observed[visible])
    absolute[state == Visibility.VISIBLE] = error
    relative[state == Visibility.VISIBLE] = error / np.maximum(observed[visible], 1e-6)
    return DepthReprojection(state, absolute, relative, selected, {
        "external": int(external.sum()),
        "z_buffer": int(behind_source.sum()),
        "reference_depth": int(behind_reference.sum()),
    })
