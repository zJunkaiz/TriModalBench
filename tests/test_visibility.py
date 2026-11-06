import unittest

import numpy as np

from trimodalbench.geometry.reprojection import project_points
from trimodalbench.geometry.visibility import Visibility, evaluate_depth


class VisibilityTests(unittest.TestCase):
    def test_z_buffer_selects_near_surface_and_classifies_other_pixels(self):
        points = np.array([[[0.8, 0, 2], [0.3, 0, 1], [8, 0, 1],
                            [0, 0, -1], [0.45, 0, 1.005], [2, 0, 1]]])
        projection = project_points(points, np.eye(3), (1, 3))
        result = evaluate_depth(projection, np.array([[1., 2., np.nan]]))
        np.testing.assert_array_equal(result.state, [[
            Visibility.OCCLUDED, Visibility.VISIBLE, Visibility.OUT_OF_BOUNDS,
            Visibility.INVALID, Visibility.VISIBLE, Visibility.INVALID,
        ]])
        np.testing.assert_allclose(result.absolute, [[np.nan, 0, np.nan, np.nan, 0.005, np.nan]])
        self.assertEqual(result.occlusion_counts, {"external": 0, "z_buffer": 1, "reference_depth": 0})

    def test_scored_depth_does_not_control_primary_visibility(self):
        points = np.array([[[0, 0, 2], [2, 0, 2], [4, 0, 2], [6, 0, 2]]])
        projection = project_points(points, np.eye(3), (1, 4))
        target = np.array([[1., 4., 1., 1.]])
        external = np.array([[True, True, False, True]])
        reference = np.array([[1., 4., 1., np.nan]])
        primary = evaluate_depth(projection, target, external_visible=external)
        assisted = evaluate_depth(projection, target, external_visible=external, reference_depth=reference)
        np.testing.assert_allclose(primary.absolute, [[1, 2, np.nan, 1]])
        np.testing.assert_allclose(primary.relative, [[1, 0.5, np.nan, 1]])
        np.testing.assert_array_equal(assisted.state, [[2, 3, 2, 3]])
        self.assertEqual(assisted.occlusion_counts, {"external": 1, "z_buffer": 0, "reference_depth": 1})

    def test_static_subset_excludes_other_points_from_z_buffer(self):
        projection = project_points(np.array([[[0, 0, 1], [0, 0, 2]]]), np.eye(3), (1, 1))
        result = evaluate_depth(projection, np.array([[2.]]), selected=np.array([[False, True]]))
        np.testing.assert_allclose(result.absolute, [[np.nan, 0]])
