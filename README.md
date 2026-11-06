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

## Validate Flow–Geometry on MPI Sintel

Install the dataset extra with `python -m pip install -e ".[sintel]"`.
Merge `clean`, `flow`, `invalid`, `occlusions`, `depth` and `camdata_left`
under one `training` directory using the official
[flow and images archives](https://sintel.is.tuebingen.mpg.de/downloads)
and [corrected depth and camera archive](https://sintel.is.tuebingen.mpg.de/depth).

```sh
python -m trimodalbench.sintel_eval --training data/sintel/training --sequence alley_1 --pairs 3 --output reports/alley_1
```

Each source-indexed flow file maps to the next RGB, depth and camera frame.
The JSON report summarizes the residual in pixels for each pair and gives
coverage relative to the entire source image. PNG maps use black for excluded
pixels and a fixed 0–5 px color range. The all-valid region can contain moving
objects. To get a second report on independently labeled static regions,
pass `--static-dir` with source-indexed `frame_XXXX.png` masks, using nonzero
pixels for static areas.

For the first three pairs of `alley_1`, the manually inspected static control
uses the intersection of the valid region with two wall patches. For each
source frame the mask is the union of NumPy slices
`[150:260, 275:440]` and `[110:225, 735:875]`, expressed as `[y, x]`.
For the first three pairs of `alley_2`, use `[70:290, 35:95]` and
`[160:220, 710:780]`. The rectangles were selected from the Clean RGB
images before scoring.

The same command also reports depth reprojection. The primary depth score uses
the official occlusion mask, a source-point Z-buffer and nearest-pixel target
depth sampling. The Z-buffer retains samples within 0.01 m plus 0.1% of the
closest projected depth in each pixel bucket. The target-assisted diagnostic
additionally excludes points behind the target depth by that tolerance. In
Sintel both scores use the same target depth for residuals, so the assisted
result is a visibility sensitivity check. For model outputs, an independent
target depth reference can be supplied to `evaluate_depth`.

The JSON includes counts for visible, occluded, out-of-bounds and invalid
points, plus a breakdown by occlusion source. Coverage uses all source pixels.
The occlusion ratio divides occluded points by the sum of occluded and visible
points. The visibility PNG shows those states on the source image grid.
The all-valid depth result includes independently moving objects and should
be interpreted alongside the static control.
