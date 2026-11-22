---
name: overhead-robot-pose
description: Turn a fixed overhead camera recording of a robot soccer / RCJ field into per-frame robot pose (field-frame x, y in metres, yaw in degrees, confidence) plus an annotated overlay video and a notes file, without ever inventing a value that was not measured. Also produces field calibration, coverage statistics and QA plots.
whenToUse: Use when someone hands you a top-down field recording (phone, PTZ or webcam on the ceiling) of a small robot on a painted field and asks for its trajectory, its position over time, its heading, a tracked overlay video, or a "pose.csv"/"ground truth" style dataset. Use also when asked to compare two pose sources or to re-run a recording through a learned detector. Do not use it for moving cameras, non-planar scenes, or multi-camera fusion.
metadata:
  version: "1.0.0"
  reference-rig: "Kaiser lab, CAM-02 PTZ, 1920x1080, nominal 30 fps, field 2190x1580 mm"
---

# Overhead robot pose

One recording in, three files out:

```
<outdir>/pose.csv     frame_idx,timestamp,x,y,yaw,conf   - one row per source frame
<outdir>/overlay.mp4  same resolution / frame count / duration as the source
<outdir>/notes.txt    conventions, calibration, timing, coverage, failure intervals
```

## The one rule that matters

**Never invent a pose value.** If the position cannot be measured, write `NaN` for
`x` and `y` (and `yaw` and `conf`). If the position is good but the heading is not
readable, keep `x`,`y` and write `NaN` for `yaw`. No interpolation, no frame
filling, no motion model, no "it probably went this way". A missing row is a
correct answer; a plausible wrong row is a defect that outlives you.

Consequences you must accept: coverage is reported honestly (typically 70-100% of
frames for position and much less for yaw), and you explain the gaps instead of
hiding them.

## Pipeline

Two modes share everything except how the robot is localised.

**geometric** (no model, no GPU, ~15-20 fps on one core)

```bash
python scripts/calibrate_field.py --image bg.jpg --out calib.json     # once per rig
python scripts/run_pose.py REC.mp4 out/ --calib calib.json
python scripts/gen_notes.py out/
```

**yolo** (a chassis detector proposes a region of interest; the plate fit refines it)

```bash
python scripts/infer_stream.py --weights best.pt --video REC.mp4 --out dets.jsonl
python scripts/run_yolo.py REC.mp4 dets.jsonl out/ --calib calib.json
python scripts/gen_notes.py out/
```

`run_yolo.py` raises position coverage a lot (76% -> 98% on the reference
recordings) because a learned detector still finds the robot at the field edge,
where a bare dark-blob search gives up. Where both modes see the robot they agree
to a median of 0.000 m, so the model adds coverage, not precision.

## Before you run anything

1. **Calibration.** Never reuse a calibration across cameras. `calibrate_field.py`
   needs a background frame with no robot on the mat. Look at the preview PNG it
   writes: the red quad must sit on the painted boundary. If auto-detection drifts
   (the printed scale spread exceeds ~3%), pass the corners yourself:
   `--corners "TLx,TLy;TRx,TRy;BRx,BRy;BLx,BLy"`.
   Check the scale spread and, if the rig has known marks, reproject them.
2. **Field frame.** The homography maps the painted rectangle to
   `(-HL,+HS) … (+HL,-HS)` mm, so the field frame is centred on the centre spot,
   `+x` along the long side towards the yellow goal, `+y` up in the image, yaw = 0
   along `+x`, counter-clockwise positive. Confirm this matches what the user
   means by "+x" before delivering.
3. **Height parallax.** The robot's marker is not on the floor. `--alpha` is
   `h/(Zc-h)`; the default 0.08 assumes a marker ~0.2 m up under a ~3 m camera.
   Getting this wrong biases every position outwards (or inwards) by alpha*r, i.e.
   roughly 8% of the distance from the image nadir. Measure the marker height if
   the user cares; the notes file documents what was assumed and gives the inverse
   transform.
4. **Frame rate.** Phone/PTZ recordings are usually VFR. The CSV `timestamp` is
   the real per-frame PTS read from the mp4 `stts` box; the overlay is re-encoded
   at a constant mean fps because encoders cannot express VFR. Say this in the
   notes.

## What "heading" means

The heading convention belongs to the robot, not to this script. The reference
robot carries a lavender pointer bar through its rotation axis with a black knob
at one end; **the knob end is the front**. If the knob is hidden (merged with the
black top cover), yaw is `NaN` - do not report the bare end. When you point the
skill at a different robot, replace `PoseFinder._heading` / `_ball_at_end` with
the marker's own rule and update the notes text. Ask the user to confirm one frame
by eye before you trust a whole dataset: a 180-degree systematic error looks
perfectly plausible in aggregate.

## Deliverable checklist

- [ ] `pose.csv` has exactly one row per source frame, `frame_idx` starting at 0.
- [ ] `timestamp` is monotonic and matches the container PTS (spot-check the first
      frames against `ffmpeg -vf showinfo`).
- [ ] `overlay.mp4` frame count, resolution and duration match the source
      (`cv2.VideoCapture` both, compare).
- [ ] `notes.txt` states origin, axis directions, units, yaw convention, which end
      is "forward", the pixel->metre transform, alpha and nadir, fps / frame count
      / VFR status, the failure intervals, and every caveat you know about.
- [ ] You told the user the coverage numbers and why the gaps exist.
- [ ] QA: `plot_traj.py` for a visual, `compare_pose.py` when comparing two runs;
      look for physically impossible jumps (>2-3 m/s between adjacent frames) and
      investigate before shipping.

## Known failure modes

- **Static dark blob on the mat** (shadow, cable, drain): it fits a circle well
  enough to be detected for seconds at a time at one fixed spot. Raise
  `--frac-min` (0.16 -> 0.22 on the reference data) and check the trajectory plot
  for a stationary cluster.
- **Cast shadow / purple reflections** near the robot: the pointer-search gates on
  the *longest continuous* colour run and rejects ambiguous opposites; keep that
  gate rather than trusting a single colour mask.
- **Half a robot / robot off the mat**: expected NaN. Do not "improve" coverage by
  relaxing the fit thresholds; you will start reporting the shadow.
- **Hand-held calibration segment** at the start of many recordings: the robot is
  tilted and carried, so the pose there is meaningless even when it is detected.
  Say so in the notes.
- **A learner detector labelling the pointer knob as "Ball"**: if you fuse a
  `Ball` class into the pipeline, gate it by geometry (it must sit on the pointer
  bar) or you will ship a ball track that follows the robot's own heading marker.

## Reference material

- `references/conventions.md` - frames, units, yaw, parallax maths, PTS extraction.
- `references/calibration.md` - how the homography is built and validated.
- `references/tuning.md` - every knob, what it trades off, what it broke.
- `references/yolo-hybrid.md` - the RCJ detector, inference over SSH on a Windows
  box, what it is good and bad at.

## Scripts

| script | role |
|---|---|
| `calibrate_field.py` | painted-line homography, preview + corner override |
| `run_pose.py` | geometric pose + overlay + stats |
| `run_yolo.py` | chassis-detector + plate refinement |
| `infer_stream.py` | streaming YOLO inference to JSONL (one line per frame) |
| `gen_notes.py` | notes.txt in zh or en from stats.json + pose.csv |
| `compare_pose.py` | agreement between two pose.csv files |
| `plot_traj.py` | field trajectory plot with sampled headings |
| `get_ffmpeg.py` | fetch a static ffmpeg with libx264 when the box has none |
