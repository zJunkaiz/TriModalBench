from dataclasses import dataclass

import numpy as np

from .backproject import backproject_depth, pixel_grid
from .camera import Camera, relative_pose


@dataclass(frozen=True)
class Projection:
    pixels: np.ndarray
    depth: np.ndarray
    positive_depth: np.ndarray
    in_bounds: np.ndarray


def transform_points(points: np.ndarray, T_target_source: np.ndarray) -> np.ndarray:
    return points @ T_target_source[:3, :3].T + T_target_source[:3, 3]


def in_bounds(pixels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    u, v = np.moveaxis(pixels, -1, 0)
    tolerance = 1e-9
    return (
        (u >= -tolerance) & (u <= width - 1 + tolerance)
        & (v >= -tolerance) & (v <= height - 1 + tolerance)
    )


def project_points(
    points: np.ndarray, K: np.ndarray, image_shape: tuple[int, int]
) -> Projection:
    depth = points[..., 2]
    positive = np.isfinite(points).all(axis=-1) & (depth > 0)
    pixels = np.full((*depth.shape, 2), np.nan)
    pixels[positive] = (points[positive] @ K.T)[:, :2] / depth[positive, None]
    return Projection(pixels, depth, positive, in_bounds(pixels, image_shape))


def reproject_depth(
    depth: np.ndarray, source: Camera, target: Camera, image_shape: tuple[int, int]
) -> Projection:
    points = backproject_depth(depth, source.K)
    target_points = transform_points(points, relative_pose(source, target))
    return project_points(target_points, target.K, image_shape)


@dataclass(frozen=True)
class PairErrors:
    flow_geometry: np.ndarray
    depth_absolute: np.ndarray
    depth_relative: np.ndarray
    pixel_counts: dict[str, int]

    @property
    def coverage(self) -> float:
        return self.pixel_counts["evaluated"] / self.pixel_counts["total"]


def _sample_depth(depth: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Bilinear interpolation requiring every contributing depth to be valid."""
    height, width = depth.shape
    u = np.clip(pixels[:, 0], 0, width - 1)
    v = np.clip(pixels[:, 1], 0, height - 1)
    x, y = u.astype(int), v.astype(int)
    right, below = np.minimum(x + 1, width - 1), np.minimum(y + 1, height - 1)
    du, dv = u - x, v - y
    values = np.stack((depth[y, x], depth[y, right], depth[below, x], depth[below, right]))
    weights = np.stack(((1 - du) * (1 - dv), du * (1 - dv), (1 - du) * dv, du * dv))
    valid = np.isfinite(values) & (values > 0)
    sampled = (np.where(valid, values, 0) * weights).sum(axis=0)
    sampled[np.any(~valid & (weights > 1e-12), axis=0)] = np.nan
    return sampled


def evaluate_pair(
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    flow: np.ndarray,
    source: Camera,
    target: Camera,
    *,
    static_mask: np.ndarray,
    visible_mask: np.ndarray,
    flow_valid_mask: np.ndarray | None = None,
) -> PairErrors:
    """Evaluate adjacent forward flow on externally identified static, visible pixels.

    All masks are boolean, on the source pixel grid, and true where usable.
    Flow vectors are (du, dv) in pixels. Unevaluated map entries are NaN.
    """
    shape = source_depth.shape
    if target_depth.shape != shape or flow.shape != (*shape, 2):
        raise ValueError("Expected aligned depth [H, W] and flow [H, W, 2]")
    for mask in (static_mask, visible_mask, flow_valid_mask):
        if mask is not None and (mask.shape != shape or mask.dtype != np.bool_):
            raise ValueError("Masks must be boolean arrays with depth shape")
    projection = reproject_depth(source_depth, source, target, shape)
    flow_target = pixel_grid(shape) + flow
    valid_depth = np.isfinite(source_depth) & (source_depth > 0)
    valid_flow = np.isfinite(flow).all(axis=-1)
    if flow_valid_mask is not None:
        valid_flow &= flow_valid_mask
    candidate = valid_depth & valid_flow & static_mask & visible_mask
    candidate &= projection.in_bounds & in_bounds(flow_target, shape)
    sampled_depth = _sample_depth(target_depth, projection.pixels[candidate])
    valid = candidate.copy()
    valid[candidate] = np.isfinite(sampled_depth) & (sampled_depth > 0)
    sampled_depth = sampled_depth[valid[candidate]]
    flow_error = np.full(shape, np.nan)
    depth_error = np.full(shape, np.nan)
    relative_error = np.full(shape, np.nan)
    flow_error[valid] = np.linalg.norm(flow_target[valid] - projection.pixels[valid], axis=-1)
    depth_error[valid] = np.abs(projection.depth[valid] - sampled_depth)
    relative_error[valid] = depth_error[valid] / np.maximum(sampled_depth, 1e-6)
    return PairErrors(flow_error, depth_error, relative_error, {
        "total": source_depth.size,
        "depth_valid": int(valid_depth.sum()),
        "flow_valid": int(valid_flow.sum()),
        "static": int(static_mask.sum()),
        "visible": int(visible_mask.sum()),
        "evaluated": int(valid.sum()),
    })
