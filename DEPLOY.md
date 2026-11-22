# Deploying this skill to another agent

Two prompts. The first installs the skill on a machine; the second is the task
prompt you hand to an agent afterwards. Both are copy-paste, both assume the agent
has a shell.

## 1. Install prompt

> You are installing a reusable skill on this machine. Do this in order and stop
> on the first failure, reporting exactly what failed:
>
> 1. Clone <https://github.com/423Tech/overhead-robot-pose> into the skill root of
>    your harness, as a directory named `overhead-robot-pose`:
>    - DSH: `~/.dsh/skills/overhead-robot-pose` (or `<project>/.agents/skills/...`)
>    - any other harness: wherever it discovers `SKILL.md` bundles
>    A symlink to a clone is fine.
> 2. Check the bundle: `SKILL.md` at the top level, `scripts/`, `references/`,
>    `assets/` beside it. If the harness exposes a skill catalog, reload it and
>    confirm `overhead-robot-pose` is listed.
> 3. Install dependencies: `python -c "import cv2, numpy"`. If that fails:
>    `pip install -r requirements.txt`. If `ffmpeg` is not on PATH, run
>    `python scripts/get_ffmpeg.py`.
> 4. Read `SKILL.md` and `references/conventions.md` before doing anything else.
> 5. Smoke-test on any short field video you can find (or `--limit 60` on a longer
>    one) with the bundled reference calibration:
>    `python scripts/run_pose.py REC.mp4 out/ --limit 60 --calib assets/calib_kaiser_example.json`
>    Expect `pose.csv`, `overlay.mp4`, `stats.json`, and a `notes.txt` from
>    `python scripts/gen_notes.py out/`. Report the coverage numbers the run
>    printed.
> 6. Report: install path, whether the catalog shows the skill, dependency status,
>    smoke-test coverage, and anything that did not work.

## 2. Task prompt

> Use the `overhead-robot-pose` skill to analyse these recordings:
> `<video files or directory>`.
>
> Requirements:
>
> - One delivery folder per recording, named after the video without its
>   extension, beside the source file if it is writable, otherwise in
>   `<fallback dir>/`. Each folder contains `pose.csv`, `overlay.mp4`, `notes.txt`.
> - `pose.csv`: header exactly `frame_idx,timestamp,x,y,yaw,conf`; **one row per
>   source frame**, `frame_idx` from 0, `timestamp` = real per-frame PTS from the
>   container, `x,y` in metres in the field frame, `yaw` in degrees, `conf` 0-1
>   or NaN.
> - Never interpolate, never fill frames, never extrapolate with a motion model.
>   If the position cannot be measured, the row is NaN. If the position is good
>   but the heading is unreadable, `yaw` is NaN. Missing beats wrong.
> - `overlay.mp4`: same resolution, frame count and duration as the source, no
>   dropped or added frames, h264; it must show position and heading, and say
>   "no detection" instead of drawing a guess.
> - `notes.txt`: origin, axis directions, units, yaw convention, which end of the
>   robot is "forward", the pixel->metre transform actually used, the height
>   parallax assumption, fps / frame count / CFR or VFR, and the frame ranges that
>   failed or were occluded.
> - Calibrate per camera; never reuse another camera's calibration. Show me the
>   preview image and the scale spread before running the whole video.
> - Report at the end: frames, position coverage, yaw coverage, why the gaps
>   exist, and any frame you are unsure about. Do not summarise away the NaN rate.

## 3. Variants

**Recalibrate for a new camera**

> Before analysing, build a background frame with the mat empty, run
> `scripts/calibrate_field.py --image bg.jpg --out calib.json`, and show me
> `calib.json.preview.png` and the four px/mm scales. Do not continue until I
> confirm the red quad sits on the painted line. If auto-detection drifts, read the
> corners off the preview and pass them with `--corners`.

**Add a learned detector**

> Install ultralytics and point `scripts/infer_stream.py` at a chassis detector
> (`--weights best.pt --imgsz 1280 --conf 0.05`), then feed `dets.jsonl` to
> `scripts/run_yolo.py`. Report both coverages and the agreement between the two
> modes on the frames where both found the robot; a violation of the
> "missing beats wrong" rule is a blocker, extra coverage is not.

**Compare two pose sources**

> Run `scripts/compare_pose.py a/pose.csv b/pose.csv` and report the median and p99
> position difference, the fraction of frames disagreeing by more than 15 cm, and
> the yaw difference distribution. Investigate any disagreement above 15 cm before
> drawing conclusions.

## 4. What to demand back

An agent that used this skill should be able to answer, without re-running
anything:

- how many frames, how many with a position, how many with a heading;
- which calibration file, which alpha and nadir, and why;
- which frame ranges are missing and what was happening there;
- whether the overlay frame count matches the source;
- what it did **not** trust, and what a human should check.

If it cannot answer any of those from `notes.txt` and `stats.json`, the notes are
incomplete - fix the notes, not the answer.
