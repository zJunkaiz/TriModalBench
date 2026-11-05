"""Run the geometry sanity experiment with an analytic static plane."""

import json
from dataclasses import asdict

import numpy as np

from .geometry.camera import Camera
from .geometry.reprojection import evaluate_pair
from .temporal import delay_depth, estimate_lag


def plane_sequence(
    step_m: float = 0.22,
) -> tuple[list[np.ndarray], list[Camera], list[np.ndarray], list[np.ndarray]]:
    height, width = 24, 32
    cx, cy = (width - 1) / 2, (height - 1) / 2
    K = np.array([[32, 0, cx], [0, 32, cy], [0, 0, 1]])
    v, u = np.indices((height, width))
    pixels_from_center = np.stack((u - cx, v - cy), axis=-1)
    cameras, depths, flows, masks = [], [], [], []
    for frame in range(20):
        pose = np.eye(4)
        pose[2, 3] = frame * step_m
        cameras.append(Camera(K, pose))
        depths.append(np.full((height, width), 6.0 - frame * step_m))
    # Closed-form expansion around the principal point for forward translation.
    for depth in depths[1:]:
        flow = pixels_from_center * (step_m / depth[..., None])
        flows.append(flow)
        masks.append(
            (u + flow[..., 0] >= 0) & (u + flow[..., 0] <= width - 1)
            & (v + flow[..., 1] >= 0) & (v + flow[..., 1] <= height - 1)
        )
    return depths, cameras, flows, masks


def main() -> None:
    depths, cameras, flows, visible = plane_sequence()
    static = [np.ones_like(mask) for mask in visible]
    clean = evaluate_pair(
        depths[0], depths[1], flows[0], cameras[0], cameras[1],
        static_mask=static[0], visible_mask=visible[0],
    )
    scaled = evaluate_pair(
        1.2 * depths[0], 1.2 * depths[1], flows[0], cameras[0], cameras[1],
        static_mask=static[0], visible_mask=visible[0],
    )
    report = {}
    for name, errors in (("clean", clean), ("depth_scale_1.2", scaled)):
        report[name] = {
            "flow_geometry_mean_px": float(np.nanmean(errors.flow_geometry)),
            "flow_geometry_median_px": float(np.nanmedian(errors.flow_geometry)),
            "flow_geometry_p95_px": float(np.nanpercentile(errors.flow_geometry, 95)),
            "depth_relative_median": float(np.nanmedian(errors.depth_relative)),
            "coverage": errors.coverage,
            "pixel_counts": errors.pixel_counts,
        }
    report["lag_search"] = {
        delay: asdict(estimate_lag(delay_depth(depths, delay), cameras, flows, static, visible))
        for delay in range(4)
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
