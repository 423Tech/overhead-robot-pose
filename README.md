# overhead-robot-pose

Recover a robot's pose from a fixed overhead camera recording of a painted field:
per-frame `x, y, yaw, conf`, an annotated overlay video, and a notes file that
documents every assumption. No GPU required for the geometric mode; an optional
YOLO chassis detector raises coverage on the hard frames.

Built for RoboCup Junior / RCJ Soccer Open field recordings, first used on the
Kaiser lab PTZ camera (1920x1080, ~3 m above a 2190x1580 mm field).

```
pose.csv     frame_idx,timestamp,x,y,yaw,conf      one row per source frame
overlay.mp4  identical resolution / frame count / duration
notes.txt    conventions, calibration, timing, coverage, failure intervals
```

## Design rule

**Missing beats wrong.** A frame whose pose cannot be measured gets `NaN`, never an
interpolated or guessed value. A readable position with an unreadable heading keeps
`x,y` and sets `yaw` to `NaN`. Coverage is reported as-is and the gaps are listed in
`notes.txt`.

## Quick start

```bash
pip install -r requirements.txt          # numpy, opencv-python
python scripts/get_ffmpeg.py             # only if ffmpeg is not on PATH

# 1. calibrate once per camera (needs a background frame with the mat empty)
python scripts/calibrate_field.py --image bg.jpg --out calib.json
#    look at calib.json.preview.png before trusting it

# 2. pose for a whole recording
python scripts/run_pose.py REC.mp4 out/ --calib calib.json
python scripts/gen_notes.py out/                       # notes.txt (zh: --lang zh)
python scripts/plot_traj.py out/pose.csv traj.png       # optional QA plot
```

With a detector (stronger coverage at the field edge):

```bash
python scripts/infer_stream.py --weights best.pt --video REC.mp4 --out dets.jsonl
python scripts/run_yolo.py REC.mp4 dets.jsonl out/ --calib calib.json
python scripts/gen_notes.py out/
```

`assets/calib_kaiser_example.json` is the validated calibration of the reference
rig - reuse it only for that exact camera, never for a different one.

## What is in here

| path | contents |
|---|---|
| `SKILL.md` | the agent-facing skill: workflow, rules, checklist |
| `scripts/` | the pipeline (calibration, pose, overlay, notes, QA) |
| `references/` | conventions and maths, calibration method, tuning notes, YOLO notes |
| `assets/` | example calibration for the reference rig |

## Results on the reference recordings

| recording | frames | position | yaw |
|---|---|---|---|
| 21-18-33 | 22 161 | 76.2% geometric / 98.2% hybrid | 10.7% / 12.9% |
| 21-40-58 | 20 411 | 71.3% geometric / 98.0% hybrid | 24.8% / 26.0% |

Yaw coverage is deliberately low: it requires the heading marker's front to be
readable, and the pipeline refuses to guess when it is not.

## License

MIT - see `LICENSE`.
