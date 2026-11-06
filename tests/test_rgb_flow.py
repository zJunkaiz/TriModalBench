import unittest

import numpy as np

from trimodalbench.geometry.camera import Camera
from trimodalbench.geometry.reprojection import reproject_depth
from trimodalbench.geometry.backproject import pixel_grid
from trimodalbench.rgb_flow import CHARBONNIER_EPSILON, compute_rgb_flow_consistency


class RGBFlowTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((5, 6))
        self.target = np.stack((x / 12, y / 10, (x + y) / 20), axis=-1)
        self.source = self.target + [0.5 / 12, 0.25 / 10, 0.75 / 20]
        self.flow = np.broadcast_to([0.5, 0.25], (5, 6, 2)).copy()

    def test_forward_subpixel_warp(self):
        result = compute_rgb_flow_consistency(self.source, self.target, self.flow)
        valid = result.valid_mask
        self.assertEqual(valid.sum(), 20)
        np.testing.assert_allclose(result.warped_rgb[valid], self.source[valid])
        np.testing.assert_allclose(result.difference_map[valid], 0, atol=1e-15)
        np.testing.assert_allclose(result.error_map[valid], CHARBONNIER_EPSILON)
        np.testing.assert_array_equal(np.isfinite(result.identity_error_map), valid)
        self.assertTrue(np.all(result.identity_error_map[valid] > result.error_map[valid]))

    def test_zero_flow_matches_identity_on_moving_input(self):
        result = compute_rgb_flow_consistency(self.source, self.target, np.zeros_like(self.flow))
        np.testing.assert_array_equal(result.error_map, result.identity_error_map)
        self.assertTrue(np.all(result.error_map > CHARBONNIER_EPSILON))

    def test_wrong_direction_is_worse_on_common_support(self):
        good = compute_rgb_flow_consistency(self.source, self.target, self.flow)
        wrong = compute_rgb_flow_consistency(self.source, self.target, -self.flow)
        common = good.valid_mask & wrong.valid_mask
        self.assertTrue(common.any())
        self.assertTrue(np.all(wrong.error_map[common] > wrong.identity_error_map[common]))
        self.assertTrue(np.all(wrong.error_map[common] > good.error_map[common]))

    def test_invalid_and_occluded_pixels_are_excluded(self):
        visible = np.ones((5, 6), dtype=bool)
        flow_valid = visible.copy()
        visible[0, 0] = False
        flow_valid[0, 1] = False
        result = compute_rgb_flow_consistency(
            self.source, self.target, self.flow, flow_valid, ~visible,
        )
        self.assertEqual(result.valid_mask.sum(), 18)
        self.assertTrue(np.isnan(result.warped_rgb[0, :2]).all())

    def test_out_of_bounds_and_nonfinite_flow_are_excluded(self):
        self.flow[0, 0] = [-0.25, 0]
        self.flow[0, 1] = [np.nan, 0]
        result = compute_rgb_flow_consistency(self.source, self.target, self.flow)
        self.assertFalse(result.valid_mask[-1].any())
        self.assertFalse(result.valid_mask[:, -1].any())
        self.assertFalse(result.valid_mask[0, :2].any())

    def test_brightness_jump_with_correct_flow(self):
        result = compute_rgb_flow_consistency(self.source, self.target + 0.1, self.flow)
        np.testing.assert_allclose(result.error_map[result.valid_mask], np.hypot(0.1, CHARBONNIER_EPSILON))

    def test_moving_object_can_have_low_rgb_error_and_high_geometry_error(self):
        source = np.zeros((5, 8, 3), dtype=np.uint8)
        target = source.copy()
        source[1:4, 1:3] = 180
        target[1:4, 3:5] = 180
        flow = np.zeros((5, 8, 2))
        flow[1:4, 1:3, 0] = 2
        occluded = np.zeros((5, 8), dtype=bool)
        occluded[1:4, 3:5] = True
        rgb = compute_rgb_flow_consistency(source, target, flow, occlusion_mask=occluded)
        camera = Camera(np.eye(3), np.eye(4))
        grid = pixel_grid(source.shape[:2])
        projection = reproject_depth(np.ones(source.shape[:2]), camera, camera, source.shape[:2])
        geometry_error = np.linalg.norm(grid + flow - projection.pixels, axis=-1)
        np.testing.assert_allclose(rgb.error_map[1:4, 1:3], CHARBONNIER_EPSILON)
        np.testing.assert_allclose(geometry_error[1:4, 1:3], 2)
