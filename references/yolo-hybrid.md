# The YOLO chassis detector (hybrid mode)

## What it is

A YOLOv8s five-class detector trained on human-reviewed RCJ Soccer Open frames:

```
0 Ball   1 BlueGoal   2 YellowGoal   3 Chassis   4 Dribbler
```

Reference weights: `G:\RCJReview-v8\runs\yolov8s-round2-v1-20260925-135033\weights\best.pt`
(training view on `D:\RCJReview-v8`, ultralytics, val mAP50 0.73 after the single
logged epoch - the run stopped early, so more epochs should help).

Nothing about the pipeline is model-specific: any detector that emits a `chassis`
(or "robot") box per frame can feed `run_yolo.py` through `infer_stream.py`'s JSONL
format.

## Why it is only a region proposer

Measured on the two reference recordings (22 161 + 20 411 frames):

| | geometric only | + YOLO |
|---|---|---|
| position coverage | 76.2% / 71.3% | **98.2% / 98.0%** |
| yaw coverage | 10.7% / 24.8% | 12.9% / 26.0% |
| plate refined inside the box | - | 86.4% / 75.7% |

- On frames where both modes find the robot, the position differs by **0.000 m
  median** (p99 1-1.6 cm). The model does not make the pose more precise; it makes
  it exist more often, mainly at the field edge where a bare dark-blob search has
  nothing to lock onto.
- The model **never** detected the `Dribbler` class on this robot (0 hits in 42k
  frames): from directly above there is no visible roller. So the model cannot
  supply the heading; the pointer-knob logic still does.
- The chassis box centre is a usable fallback (median 1.4 cm, p90 5.2 cm from the
  plate centre) **because the box is tight**. If you widen the box or include the
  pointer, this fallback degrades fast - always keep the plate refinement first.

## Running the detector

On a machine with ultralytics + CUDA:

```bash
python scripts/infer_stream.py --weights best.pt --video REC.mp4 --out dets.jsonl \
       --imgsz 1280 --conf 0.05 --iou 0.55
```

`infer_stream.py` streams the video through `model.predict(source=..., stream=True)`
so decoding and batching happen inside ultralytics; on an RTX 3090 this ran at
43-45 fps at imgsz 1280 (a naive `predict(list_of_frames)` loop was 2.3 fps, ~20x
slower, because each call rebuilds the dataloader).

Output is one JSON object per **source** frame, in order:

```json
{"f": 1234, "d": [[3, 0.87, 1392.0, 439.0, 1576.0, 583.0], [0, 0.52, ...]]}
```

`f` must equal the frame index; `run_yolo.py` pairs line *i* with frame *i* and
stops at the first mismatch, so never filter lines after the fact.

## Driving a Windows box over SSH (what we actually did)

The reference detector lives on a Windows workstation. Notes that saved time:

- the box has several Python environments; the working one was a project venv
  (`G:\Github\Project-Levelupper-rcj\.venv-yolo`), not the system Python;
- the default shell is PowerShell and quoting a path with spaces through
  `ssh host 'cmd /c "..."'` mangles it - copy the videos to space-free names, or
  upload a `.bat`/`.ps1` and run that;
- `scp` the inference script to the box and run it there, rather than trying to
  build a giant one-liner;
- write the JSONL with a large buffer (1 MB); reading the file while it is still
  buffered looks like "nothing is happening";
- `wmic` is gone on recent Windows - use PowerShell cmdlets or `nvidia-smi`.

## The Ball class is not a ball tracker (on this robot)

`Ball` detections on the reference recordings are dominated by the robot's own
black pointer knob: the model confidently (0.5-0.8) boxes the knob as a ball, e.g.
frame 14730 of video 1 at (1557, 720). Median confidence of the class was 0.14
(video 1) and 0.31 (video 2), i.e. mostly low-grade noise. Consequences:

- do not ship a ball trajectory built from this class without filtering;
- a *geometric* gate helps: keep a `Ball` box only if it is within ~25 px of the
  pointer bar's line, at 1.0-2.6 plate radii from the axis - then treat it as
  evidence for *which end carries the knob*, which is exactly what `--fuse` does;
- a real game ball in front of the dribbler also satisfies that gate, which is
  fine: a ball the robot is about to dribble is in front of the robot.

`--fuse` is off by default because it was not validated broadly enough to change
delivered data. Turn it on only if you can eyeball a few hundred frames.
