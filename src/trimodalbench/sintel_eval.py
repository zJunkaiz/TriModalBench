"""Generate frame-level cross-modal consistency reports for Sintel."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry.backproject import pixel_grid
from .geometry.reprojection import Projection, in_bounds, reproject_depth
from .geometry.visibility import DepthReprojection, Visibility, evaluate_depth
from .rgb_flow import CHARBONNIER_EPSILON, RGBFlowResult, compute_rgb_flow_consistency
from .sintel import FramePair, iter_frame_pairs


def flow_geometry(
    pair: FramePair, projection: Projection | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return residuals on all usable pixels and on independently masked static pixels."""
    shape = pair.source_depth.shape
    if projection is None:
        projection = reproject_depth(pair.source_depth, pair.source_camera, pair.target_camera, shape)
    flow_pixels = pixel_grid(shape) + pair.flow
    valid = pair.flow_valid & pair.visible & projection.in_bounds & in_bounds(flow_pixels, shape)
    errors = np.full(shape, np.nan)
    errors[valid] = np.linalg.norm(flow_pixels[valid] - projection.pixels[valid], axis=-1)
    static_errors = None if pair.static is None else np.where(pair.static, errors, np.nan)
    return errors, static_errors


def summarize(errors: np.ndarray, unit: str = "px") -> dict:
    valid = errors[np.isfinite(errors)]
    return {
        f"mean_{unit}": float(valid.mean()) if valid.size else None,
        f"median_{unit}": float(np.median(valid)) if valid.size else None,
        f"p95_{unit}": float(np.percentile(valid, 95)) if valid.size else None,
        "valid_pixels": int(valid.size),
        "source_pixels": int(errors.size),
        "coverage": valid.size / errors.size,
    }


def save_map(path: Path, errors: np.ndarray, ceiling: float) -> None:
    valid = np.isfinite(errors)
    colors = np.zeros((*errors.shape, 3), dtype=np.uint8)
    level = np.clip(errors[valid] / ceiling, 0, 1)
    colors[valid] = np.column_stack((255 * level, np.full(level.shape, 32), 255 * (1 - level)))
    Image.fromarray(colors).save(path)


def summarize_rgb_flow(result: RGBFlowResult, region: np.ndarray | None = None) -> dict:
    error, identity = result.error_map, result.identity_error_map
    if region is not None:
        error, identity = np.where(region, error, np.nan), np.where(region, identity, np.nan)
    summary = summarize(error, "intensity")
    baseline = summarize(identity, "intensity")["mean_intensity"]
    summary["identity_mean_intensity"] = baseline
    summary["alignment_gain"] = (
        None if baseline is None
        else (baseline - summary["mean_intensity"]) / (baseline + CHARBONNIER_EPSILON)
    )
    return summary


def summarize_depth(result: DepthReprojection) -> dict:
    counts = {
        state.name.lower(): int(np.count_nonzero(result.selected & (result.state == state)))
        for state in Visibility
    }
    classified = counts["visible"] + counts["occluded"]
    return {
        "absolute": summarize(result.absolute, "m"),
        "relative": summarize(result.relative, "ratio"),
        "selected_pixels": int(result.selected.sum()),
        "visibility_counts": counts,
        "occlusion_counts": result.occlusion_counts,
        "visible_coverage": counts["visible"] / result.state.size,
        "occlusion_ratio": counts["occluded"] / classified if classified else None,
    }


def depth_report(
    pair: FramePair, projection: Projection, output: Path, name: str,
    *, selected: np.ndarray | None,
) -> dict:
    primary = evaluate_depth(projection, pair.target_depth, selected=selected, external_visible=pair.visible)
    diagnostic = evaluate_depth(
        projection, pair.target_depth, selected=selected,
        external_visible=pair.visible, reference_depth=pair.target_depth,
    )
    save_map(output / f"{name}_depth.png", primary.absolute, 0.1)
    palette = np.array([[38, 38, 38], [90, 120, 255], [255, 130, 40], [40, 210, 110]], dtype=np.uint8)
    colors = palette[primary.state]
    colors[~primary.selected] = 0
    Image.fromarray(colors).save(output / f"{name}_visibility.png")
    return {
        "primary": summarize_depth(primary),
        "target_depth_assisted_diagnostic": summarize_depth(diagnostic),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True, help="Merged Sintel training directory")
    parser.add_argument("--sequence", default="alley_1")
    parser.add_argument("--start", type=int)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--static-dir", type=Path, help="Source-frame PNG masks. Nonzero means static.")
    parser.add_argument(
        "--moving-roi", nargs=5, type=int, metavar=("FRAME", "X0", "Y0", "X1", "Y1"),
        help="Compare a manually selected moving region at one source frame (end exclusive).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ceiling_px = 5.0
    rgb_ceiling = 0.1
    frames = []
    for pair in iter_frame_pairs(
        args.training, args.sequence, start=args.start, count=args.pairs, static_dir=args.static_dir,
    ):
        projection = reproject_depth(
            pair.source_depth, pair.source_camera, pair.target_camera, pair.target_depth.shape,
        )
        errors, static_errors = flow_geometry(pair, projection)
        rgb = compute_rgb_flow_consistency(
            pair.source_rgb, pair.target_rgb, pair.flow, pair.flow_valid, ~pair.visible,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        name = f"frame_{pair.source_id:04d}"
        row = {
            "source_frame": pair.source_id,
            "target_frame": pair.target_id,
            "official_flow_valid_pixels": int(pair.flow_valid.sum()),
            "official_visible_pixels": int(pair.visible.sum()),
            "all_valid": summarize(errors),
            "static": None,
            "rgb_flow": {"all_valid": summarize_rgb_flow(rgb), "static": None},
            "depth_reprojection": {
                "all_valid": depth_report(pair, projection, args.output, f"{name}_all", selected=None),
                "static": None,
            },
        }
        save_map(args.output / f"{name}_error.png", rgb.error_map, rgb_ceiling)
        Image.fromarray(np.rint(np.nan_to_num(rgb.difference_map) * 255).astype(np.uint8)).save(
            args.output / f"{name}_rgb_difference.png"
        )
        Image.fromarray(rgb.valid_mask.astype(np.uint8) * 255).save(
            args.output / f"{name}_rgb_validity.png"
        )
        save_map(args.output / f"{name}_all.png", errors, ceiling_px)
        if static_errors is not None:
            row["static"] = summarize(static_errors)
            row["rgb_flow"]["static"] = summarize_rgb_flow(rgb, pair.static)
            row["static_mask_pixels"] = int(pair.static.sum())
            row["depth_reprojection"]["static"] = depth_report(
                pair, projection, args.output, f"{name}_static", selected=pair.static,
            )
            save_map(args.output / f"{name}_static.png", static_errors, ceiling_px)
        if args.moving_roi and pair.source_id == args.moving_roi[0]:
            _, x0, y0, x1, y1 = args.moving_roi
            height, width = rgb.valid_mask.shape
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                parser.error("--moving-roi must be a nonempty rectangle on the source image")
            region = np.zeros_like(rgb.valid_mask)
            region[y0:y1, x0:x1] = True
            row["moving_roi"] = {
                "xyxy": [x0, y0, x1, y1],
                "flow_geometry": summarize(np.where(region, errors, np.nan)),
                "rgb_flow": summarize_rgb_flow(rgb, region),
            }
        frames.append(row)
    if args.moving_roi and not any("moving_roi" in row for row in frames):
        parser.error("--moving-roi frame is outside the evaluated source frames")
    report = {
        "sequence": args.sequence,
        "metric": "flow_geometry",
        "metrics": ["flow_geometry", "depth_reprojection", "rgb_flow"],
        "rgb_flow_heatmap": {
            "black": "excluded", "blue_intensity": 0, "red_intensity_at_or_above": rgb_ceiling,
        },
        "all_valid_includes_independently_moving_objects": True,
        "heatmap": {"black": "invalid", "blue_px": 0, "red_px_at_or_above": ceiling_px},
        "depth_heatmap": {"black": "excluded", "blue_m": 0, "red_m_at_or_above": 0.1},
        "visibility_colors": {
            "black": "outside selected region", "gray": "invalid", "blue": "out_of_bounds",
            "orange": "occluded", "green": "visible",
        },
        "frames": frames,
    }
    serialized = json.dumps(report, indent=2, allow_nan=False)
    (args.output / "report.json").write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
