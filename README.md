# TriModalBench

Pose-aware geometry diagnostics for trajectory-aligned multimodal video predictions.
The initial experiment checks whether metric Z-depth and adjacent forward optical flow
agree with known camera motion on visible static surfaces.

## Run the geometry experiment

```sh
python -m pip install -e .
python -m trimodalbench.demo
python -m unittest discover -s tests -v
```

The demo uses a static plane and a forward-moving camera. Ground-truth flow comes
from a closed-form expansion around the principal point. The JSON includes clean
and 20% depth-scale perturbation results, plus a lag sweep for injected delays
of 0 through 3 frames. A positive lag reads a later depth index while keeping
the forward flow and both camera poses fixed.

## Geometry contract

- Depth is a metric Z-depth image of shape `[H, W]`. Nonpositive and nonfinite values are invalid.
- Flow is a source-grid array of shape `[H, W, 2]` with pixel offsets `(du, dv)` for the next frame.
- Camera coordinates use X right, Y down and Z forward. Poses map camera coordinates to world coordinates.
- Static and visibility masks are supplied on the source grid. True means the point is eligible for evaluation.

`evaluate_pair` returns pixelwise flow-to-geometry and depth reprojection error
maps, counts and coverage. It samples target depth at the projected location.
The caller provides visibility independently of the evaluated depth so that
large errors cannot automatically hide behind a depth-derived occlusion filter.

`estimate_lag` compares candidates on a shared set of frames and valid pixels.
It uses median flow-to-geometry error to rank candidates and reports median
relative depth error separately. The defaults require 10% shared coverage,
two supported pairs and a 0.01 px gap between the two best offsets. These are
prototype settings to calibrate on real sequences. Camera translation and a
separated score minimum are required for a lag estimate.

The present validation covers synthetic geometry. Real-sequence validation
requires a dataset adapter and independently defined static and visibility masks.
