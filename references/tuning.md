# Tuning, and what each knob cost us

Defaults are tuned on one rig: 1920x1080 PTZ ~3 m above a 2190x1580 mm RCJ field,
robot top plate ~55 px radius. Re-check them if the resolution or the camera height
changes; most are expressed in pixels, not millimetres.

## `pose_core.PoseFinder`

| knob | default | meaning | what happens if you get it wrong |
|---|---|---|---|
| `search` | `(170,0,1750,1080)` | mat bbox, px | too small: robots at the edge vanish; too large: goal boards and floor clutter become candidates |
| `alpha` | 0.08 | marker parallax | 0.03 error = ~3 cm radial bias at the field edge; never changes heading |
| `nadir` | (57,0) mm | parallax centre | same as above; measure it as `H @ (w/2,h/2)` |
| `frac_min` (DISC_FRAC_MIN) | 0.22 | fraction of the 64 radial rays that must find a plate edge | 0.16 lets a static shadow blob through (it scored 0.19); 0.30 starts dropping genuine edge-of-frame robots |
| `resid_max` (DISC_RESID_MAX) | 4.0 px | median circle residual | above ~5 px the "circle" is not a circle any more |
| `DISC_R` | 42-100 px | plausible plate radius | must bracket your marker at your resolution |
| `DISC_AREA` | 2200-32000 px | dark blob area | a lidar plate is ~9000 px at 55 px radius |

`detect()` returns `ok`, `cx`, `cy`, `r`, `yaw_px` (unit vector or None), `axis`
(the pointer direction even when the knob test failed - useful for fusing other
evidence), `conf`, `frac`, `resid`.

## Pointer/heading gates in `_heading`

- 96 angular bins, walk outward from `1.10 * R`, allow 5-sample gaps;
- the winning run must start within `r0 + 32 px` of the plate rim;
- the run must be at least 50 px long **and** beat the opposite direction by 10 px.

Colour alone is not enough: the bar swings from bright lavender to almost black
with the lighting, the plate has purple reflections, and the cast shadow is a
second purple-ish streak. The run-length profile plus the opposite-direction gate
is what made this stable. Removing the gate produced confident wrong answers, not
lower coverage.

## `_ball_at_end`

The pointer carries a black knob at the front end. Past the plate rim the knob is
a compact dark blob with mat on both sides (checked at three parallel offsets,
dark threshold 105, run >= 6 samples, >= 4 bright samples before, >= 3 after).
The robot body is one long dark stretch and fails the "bright on both sides" test,
which is why the test asks for both sides instead of just darkness.

If exactly one side has the knob, that side is the front. If neither or both, yaw
is `NaN`. That asymmetry is deliberate; do not "fix" it by falling back to the
longer colour run.

Known limit: when the knob is over the white line, or merged with the black top
cover, the test fails and yaw is dropped. Reference coverage: 12.9% of frames
(video 1) and 26.0% (video 2). That is the price of never reporting a 180-degree
error.

## Overlay / encoding

- raw BGR frames piped into ffmpeg: `-c:v libx264 -preset veryfast -crf 20
  -pix_fmt yuv420p -movflags +faststart`;
- constant mean fps, frame count and duration identical to the source;
- drawing: green circle = marker, cross = rotation axis, red arrow = yaw; in YOLO
  mode a green box is the detector box, an orange box means the fallback position;
- the HUD prints frame index, timestamp and alpha so a still frame is
  self-describing in a bug report.

## Quality checks worth automating

- `plot_traj.py`: a stationary cluster or a long straight line across the field is
  a detection artefact, not motion.
- `compare_pose.py`: when two pipelines share a detector, agreement should be
  exactly 0.000 m on the shared frames. Any disagreement means the ROI or the
  candidate selection changed something - investigate before shipping.
- impossible jumps: compute speed between consecutive rows; > 2-3 m/s on a small
  robot means a bad row. On the reference data 0.04-0.7% of rows show this, all in
  the hand-held calibration segment or at the field edge.
