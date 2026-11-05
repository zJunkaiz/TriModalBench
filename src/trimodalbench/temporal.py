from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .geometry.camera import Camera
from .geometry.reprojection import evaluate_pair


@dataclass(frozen=True)
class LagScore:
    flow_geometry_median_px: float
    depth_relative_median: float


@dataclass(frozen=True)
class LagResult:
    lag: int | None
    reason: str | None
    scores: dict[int, LagScore]
    coverage: float
    pair_count: int


def delay_depth(depths: Sequence[np.ndarray], frames: int) -> list[np.ndarray | None]:
    """Delay depth by a nonnegative frame count, leaving the prefix missing."""
    if not 0 <= frames <= len(depths):
        raise ValueError("Delay must be between zero and the sequence length")
    return [None] * frames + list(depths[:len(depths) - frames])


def estimate_lag(
    depths: Sequence[np.ndarray | None],
    cameras: Sequence[Camera],
    flows: Sequence[np.ndarray],
    static_masks: Sequence[np.ndarray],
    visible_masks: Sequence[np.ndarray],
    *,
    radius: int = 4,
    minimum_coverage: float = 0.1,
    minimum_gap_px: float = 0.01,
) -> LagResult:
    """Positive lag reads depth from a later index, with flow and pose fixed."""
    count = len(depths)
    if (
        count < 2 or len(cameras) != count
        or any(len(items) != count - 1 for items in (flows, static_masks, visible_masks))
    ):
        raise ValueError("Expected N poses and depths, and N-1 forward flows and mask pairs")
    if not 1 <= radius or not 0 < minimum_coverage <= 1 or not minimum_gap_px > 0:
        raise ValueError("Invalid lag search thresholds")
    offsets = range(-radius, radius + 1)
    flow_samples: dict[int, list[np.ndarray]] = {offset: [] for offset in offsets}
    depth_samples: dict[int, list[np.ndarray]] = {offset: [] for offset in offsets}
    compared = total = pair_count = 0
    has_translation = False
    for frame in range(radius, count - radius - 1):
        # Use the same time window and the same valid pixels for every offset.
        if any(depth is None for depth in depths[frame - radius:frame + radius + 2]):
            continue
        source, target = cameras[frame:frame + 2]
        pairs = {}
        for offset in offsets:
            source_depth, target_depth = depths[frame + offset:frame + offset + 2]
            assert source_depth is not None and target_depth is not None
            pairs[offset] = evaluate_pair(
                source_depth, target_depth, flows[frame], source, target,
                static_mask=static_masks[frame], visible_mask=visible_masks[frame],
            )
        common = np.logical_and.reduce([
            np.isfinite(pair.flow_geometry) & np.isfinite(pair.depth_relative)
            for pair in pairs.values()
        ])
        total += common.size
        sample_count = int(common.sum())
        if sample_count == 0:
            continue
        compared += sample_count
        pair_count += 1
        translation = target.T_world_camera[:3, 3] - source.T_world_camera[:3, 3]
        has_translation |= bool(np.linalg.norm(translation) > 1e-9)
        for offset, pair in pairs.items():
            flow_samples[offset].append(pair.flow_geometry[common])
            depth_samples[offset].append(pair.depth_relative[common])

    coverage = compared / total if total else 0.0
    scores = {
        offset: LagScore(
            float(np.median(np.concatenate(flow_samples[offset]))),
            float(np.median(np.concatenate(depth_samples[offset]))),
        )
        for offset in offsets if compared
    }
    if coverage < minimum_coverage or pair_count < 2:
        return LagResult(None, "Insufficient shared coverage", scores, coverage, pair_count)
    if not has_translation:
        return LagResult(None, "Insufficient geometric motion", scores, coverage, pair_count)
    ranked = sorted(scores, key=lambda offset: scores[offset].flow_geometry_median_px)
    best, second = ranked[:2]
    if scores[second].flow_geometry_median_px - scores[best].flow_geometry_median_px < minimum_gap_px:
        return LagResult(None, "Lag candidates are insufficiently separated", scores, coverage, pair_count)
    return LagResult(best, None, scores, coverage, pair_count)
