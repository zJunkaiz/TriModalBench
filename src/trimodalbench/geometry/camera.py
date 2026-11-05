from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    """OpenCV pinhole camera, with a camera-to-world pose in meters."""

    K: np.ndarray
    T_world_camera: np.ndarray

    def __post_init__(self) -> None:
        K = np.array(self.K, dtype=np.float64, copy=True)
        pose = np.array(self.T_world_camera, dtype=np.float64, copy=True)
        if (
            K.shape != (3, 3)
            or not np.isfinite(K).all()
            or not np.allclose(K[2], [0, 0, 1])
            or K[0, 0] <= 0
            or K[1, 1] <= 0
            or not np.isclose(K[1, 0], 0)
        ):
            raise ValueError("K must be a finite pinhole intrinsic matrix")
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
        ):
            raise ValueError("Pose must be a homogeneous 4x4 matrix")
        rotation = pose[:3, :3]
        if not (
            np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
        ):
            raise ValueError("Pose rotation must be orthonormal with determinant +1")
        K.setflags(write=False)
        pose.setflags(write=False)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "T_world_camera", pose)


def relative_pose(source: Camera, target: Camera) -> np.ndarray:
    """Transform points from the source camera into the target camera."""
    return np.linalg.solve(target.T_world_camera, source.T_world_camera)
