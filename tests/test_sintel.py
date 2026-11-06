from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None
else:
    from trimodalbench.sintel import iter_frame_pairs
    from trimodalbench.sintel_eval import flow_geometry, summarize


@unittest.skipIf(Image is None, "Install the sintel extra to test dataset integration")
class SintelTests(unittest.TestCase):
    def test_pair_alignment_pose_direction_and_mask_polarity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def path(modality, frame, extension):
                target = root / modality / "alley_1" / f"frame_{frame:04d}.{extension}"
                target.parent.mkdir(parents=True, exist_ok=True)
                return target

            for frame in (8, 9):
                Image.new("RGB", (4, 3)).save(path("clean", frame, "png"))
                depth = np.full((3, 4), 2, dtype="<f4")
                path("depth", frame, "dpt").write_bytes(b"PIEH" + struct.pack("<ii", 4, 3) + depth.tobytes())
                extrinsic = np.eye(4)[:3]
                extrinsic[0, 3] = -(frame - 8) * 2
                camera = np.concatenate((np.eye(3).ravel(), extrinsic.ravel())).astype("<f8")
                path("camdata_left", frame, "cam").write_bytes(b"PIEH" + camera.tobytes())
            flow = np.zeros((3, 4, 2), dtype="<f4")
            flow[..., 0] = -1
            path("flow", 8, "flo").write_bytes(b"PIEH" + struct.pack("<ii", 4, 3) + flow.tobytes())
            for modality, x in (("invalid", 1), ("occlusions", 2)):
                mask = Image.new("L", (4, 3))
                mask.putpixel((x, 0), 255)
                mask.save(path(modality, 8, "png"))

            [pair] = list(iter_frame_pairs(root, "alley_1"))
            self.assertEqual((pair.source_id, pair.target_id), (8, 9))
            np.testing.assert_allclose(pair.target_camera.T_world_camera[:3, 3], [2, 0, 0])
            errors, static = flow_geometry(pair)
            self.assertIsNone(static)
            self.assertEqual(summarize(errors)["valid_pixels"], 7)
            self.assertEqual(summarize(errors)["median_px"], 0)
            with self.assertRaisesRegex(ValueError, "INVALID_ALIGNMENT"):
                list(iter_frame_pairs(root, "alley_1", count=2))
