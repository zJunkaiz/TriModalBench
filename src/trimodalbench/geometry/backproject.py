import numpy as np


def pixel_grid(shape: tuple[int, int]) -> np.ndarray:
    v, u = np.indices(shape, dtype=np.float64)
    return np.stack((u, v), axis=-1)


def backproject_depth(
    depth: np.ndarray, K: np.ndarray, valid_mask: np.ndarray | None = None
) -> np.ndarray:
    """Back-project metric Z-depth. Invalid pixels produce NaN points."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2 or depth.size == 0:
        raise ValueError("Depth must have non-empty shape [H, W]")
    valid = np.isfinite(depth) & (depth > 0)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != depth.shape:
            raise ValueError("Depth mask must match the depth shape")
        valid &= valid_mask

    points = np.full((*depth.shape, 3), np.nan)
    pixels = pixel_grid(depth.shape)[valid]
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    rays = np.linalg.solve(K, homogeneous.T).T
    points[valid] = rays * depth[valid, None]
    return points
