import unittest

import numpy as np

from trimodalbench.demo import plane_sequence
from trimodalbench.geometry.backproject import backproject_depth, pixel_grid
from trimodalbench.geometry.camera import Camera, relative_pose
from trimodalbench.geometry.reprojection import evaluate_pair, project_points
from trimodalbench.temporal import delay_depth, estimate_lag


class GeometryTests(unittest.TestCase):
    def test_projection_round_trip_with_invalid_depth(self):
        K = np.array([[30, 0, 1.5], [0, 25, 0.5], [0, 0, 1]])
        depth = np.array([[2, 3, 0], [np.nan, np.inf, -1]])
        projected = project_points(backproject_depth(depth, K), K, depth.shape)
        valid = np.isfinite(depth) & (depth > 0)
        np.testing.assert_array_equal(projected.in_bounds, valid)
        np.testing.assert_allclose(projected.pixels[valid], pixel_grid(depth.shape)[valid])
        np.testing.assert_allclose(projected.depth[valid], depth[valid])

    def test_relative_pose_handles_translation_and_rotation(self):
        source_pose = np.eye(4)
        source_pose[:3, 3] = [1, 2, 3]
        target_pose = np.eye(4)
        target_pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        target_pose[:3, 3] = [2, 3, 4]
        source = Camera(np.eye(3), source_pose)
        target = Camera(np.eye(3), target_pose)
        point = relative_pose(source, target) @ [1, 2, 4, 1]
        np.testing.assert_allclose(point, [1, 0, 3, 1])

    def test_camera_rejects_reflection(self):
        with self.assertRaisesRegex(ValueError, "determinant"):
            Camera(np.eye(3), np.diag([1, -1, 1, 1]))

    def test_masks_and_invalid_target_depth_remove_pixels(self):
        camera = Camera(np.eye(3), np.eye(4))
        source = np.ones((3, 3))
        target = source.copy()
        target[0, 0] = np.nan
        static = np.ones_like(source, dtype=bool)
        visible = static.copy()
        flow_valid = static.copy()
        static[0, 1] = False
        visible[0, 2] = False
        flow_valid[1, 0] = False
        errors = evaluate_pair(
            source, target, np.zeros((3, 3, 2)), camera, camera,
            static_mask=static, visible_mask=visible, flow_valid_mask=flow_valid,
        )
        np.testing.assert_array_equal(
            np.isfinite(errors.depth_relative),
            [[False, False, False], [False, True, True], [True, True, True]],
        )
        self.assertEqual(errors.pixel_counts["evaluated"], 5)
        self.assertEqual(errors.coverage, 5 / 9)

    def test_clean_motion_and_scale_drift(self):
        depths, cameras, flows, visible = plane_sequence()
        static = np.ones_like(visible[0])
        results = [
            evaluate_pair(
                scale * depths[0], scale * depths[1], flows[0], cameras[0], cameras[1],
                static_mask=static, visible_mask=visible[0],
            )
            for scale in (1, 1.2)
        ]
        clean, scaled = results
        self.assertLess(np.nanmax(clean.flow_geometry), 1e-10)
        self.assertLess(np.nanmax(clean.depth_relative), 1e-10)
        self.assertGreater(np.nanmedian(scaled.flow_geometry), 0.05)
        self.assertGreater(np.nanmedian(scaled.depth_relative), 0.005)


class LagTests(unittest.TestCase):
    def test_recovers_clean_and_delayed_depth(self):
        depths, cameras, flows, visible = plane_sequence()
        static = [np.ones_like(mask) for mask in visible]
        for delay in range(4):
            with self.subTest(delay=delay):
                result = estimate_lag(delay_depth(depths, delay), cameras, flows, static, visible)
                self.assertEqual(result.lag, delay)
                self.assertGreater(result.coverage, 0.5)
                self.assertLess(result.scores[delay].flow_geometry_median_px, 1e-10)

    def test_stationary_camera_cannot_recover_delay(self):
        depths, cameras, flows, visible = plane_sequence(step_m=0)
        result = estimate_lag(delay_depth(depths, 2), cameras, flows, visible, visible)
        self.assertIsNone(result.lag)
        self.assertEqual(result.reason, "Insufficient geometric motion")

    def test_empty_common_support_returns_unreliable(self):
        depths, cameras, flows, visible = plane_sequence()
        static = [np.zeros_like(mask) for mask in visible]
        result = estimate_lag(depths, cameras, flows, static, visible)
        self.assertIsNone(result.lag)
        self.assertEqual(result.coverage, 0)
        self.assertEqual(result.scores, {})
