"""Photometric consistency on the source grid of a forward flow field."""

from dataclasses import dataclass

import numpy as np

from .geometry.backproject import pixel_grid
from .geometry.reprojection import in_bounds


CHARBONNIER_EPSILON = 1e-3


@dataclass(frozen=True)
class RGBFlowResult:
    error_map: np.ndarray
    identity_error_map: np.ndarray
    warped_rgb: np.ndarray
    difference_map: np.ndarray
    valid_mask: np.ndarray


def _normalized_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float64) / 255
    if (
        np.issubdtype(image.dtype, np.floating)
        and np.isfinite(image).all() and np.all((image >= 0) & (image <= 1))
    ):
        return image.astype(np.float64, copy=False)
    raise ValueError("RGB must be uint8 or finite floating-point values in [0, 1]")


def _sample_bilinear(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    u, v = pixels.T
    u, v = np.clip(u, 0, width - 1), np.clip(v, 0, height - 1)
    x, y = u.astype(np.intp), v.astype(np.intp)
    right, bottom = np.minimum(x + 1, width - 1), np.minimum(y + 1, height - 1)
    dx, dy = (u - x)[:, None], (v - y)[:, None]
    top = (1 - dx) * image[y, x] + dx * image[y, right]
    lower = (1 - dx) * image[bottom, x] + dx * image[bottom, right]
    return (1 - dy) * top + dy * lower


def _photometric(residual: np.ndarray) -> np.ndarray:
    return np.hypot(residual, CHARBONNIER_EPSILON).mean(axis=-1)


def compute_rgb_flow_consistency(
    rgb_source: np.ndarray,
    rgb_target: np.ndarray,
    flow_forward: np.ndarray,
    flow_valid_mask: np.ndarray | None = None,
    occlusion_mask: np.ndarray | None = None,
) -> RGBFlowResult:
    """Sample target RGB at p + flow(p). True occlusion entries are excluded.

    RGB images share a pixel grid. Invalid output pixels are NaN, and the flow
    and identity residuals always use identical support.
    """
    shape = rgb_source.shape[:2]
    if (
        rgb_source.ndim != 3 or rgb_source.shape[2] != 3 or rgb_source.size == 0
        or rgb_target.shape != rgb_source.shape or flow_forward.shape != (*shape, 2)
    ):
        raise ValueError("Expected nonempty RGB [H, W, 3] and flow [H, W, 2] on the same grid")
    for mask in (flow_valid_mask, occlusion_mask):
        if mask is not None and (mask.shape != shape or mask.dtype != np.bool_):
            raise ValueError("Masks must be boolean arrays on the source grid")
    source, target = _normalized_rgb(rgb_source), _normalized_rgb(rgb_target)
    pixels = pixel_grid(shape) + flow_forward
    valid = np.isfinite(flow_forward).all(axis=-1) & in_bounds(pixels, shape)
    if flow_valid_mask is not None:
        valid &= flow_valid_mask
    if occlusion_mask is not None:
        valid &= ~occlusion_mask
    warped = np.full(source.shape, np.nan)
    warped[valid] = _sample_bilinear(target, pixels[valid])
    residual = source[valid] - warped[valid]
    difference = np.full(source.shape, np.nan)
    difference[valid] = np.abs(residual)
    error = np.full(shape, np.nan)
    identity = np.full(shape, np.nan)
    error[valid] = _photometric(residual)
    identity[valid] = _photometric(source[valid] - target[valid])
    return RGBFlowResult(error, identity, warped, difference, valid)
