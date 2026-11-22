#!/usr/bin/env python3
"""Pose from a YOLO chassis detector + plate refinement.

    run_yolo.py VIDEO DETS.jsonl OUTDIR [--calib F] [--alpha 0.08] ...

DETS.jsonl comes from infer_stream.py: one line per source frame,
{"f": idx, "d": [[cls, conf, x1, y1, x2, y2], ...]} in source-video pixels.

Why bother: a learned detector finds the robot far more often than a bare
dark-blob search (98% vs 76% of frames on our recordings), especially at the
field edge.  Its box is only used as a region of interest and as the fallback
position; the precise rotation axis still comes from the plate fit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
from pose_core import PoseFinder, ALPHA, NADIR_MM, IMG_SEARCH, DISC_FRAC_MIN, DISC_RESID_MAX

CHASSIS, BALL, BLUE, YELLOW, DRIBBLER = 3, 0, 1, 2, 4
ALPHA_BOX_DEFAULT = 0.0514   # whole-body bbox centre (field ops calibrated value)


def pick(dets, cls):
    """Best box of a class; chassis boxes must look like the robot (size prior)."""
    best = None
    for d in dets:
        if d[0] != cls:
            continue
        w = d[4] - d[2]
        h = d[5] - d[3]
        plausible = True
        if cls == CHASSIS:
            plausible = (90 <= w <= 330) and (70 <= h <= 300) and (0.55 <= w / max(h, 1e-6) <= 2.0)
        key = (1 if plausible else 0, d[1], w * h)
        if best is None or key > best[0]:
            best = (key, d)
    return best[1] if best else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("dets")
    ap.add_argument("outdir")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--alpha", type=float, default=ALPHA, help="plate parallax")
    ap.add_argument("--alpha-box", type=float, default=ALPHA_BOX_DEFAULT,
                    help="parallax for the bbox-centre fallback")
    ap.add_argument("--nadir", default="%g,%g" % NADIR_MM)
    ap.add_argument("--search", default="%d,%d,%d,%d" % IMG_SEARCH)
    ap.add_argument("--frac-min", type=float, default=DISC_FRAC_MIN)
    ap.add_argument("--resid-max", type=float, default=DISC_RESID_MAX)
    ap.add_argument("--ball-conf", type=float, default=0.40)
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--fuse", action="store_true",
                    help="let a model Ball on the pointer's end decide the heading")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    calib = common.resolve_calib(args.calib)
    H = common.load_homography(calib)
    nadir = tuple(float(v) for v in args.nadir.split(","))
    search = tuple(int(float(v)) for v in args.search.split(","))

    ts, dur_ticks, deltas, pts = common.video_timing(args.video)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("cannot open video: %s" % args.video)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if len(pts) != nframes:
        print("WARN: pts %d != frames %d" % (len(pts), nframes))
        nframes = min(nframes, len(pts))
    duration = pts[nframes - 1] + deltas[nframes - 1] / float(ts)
    fps = nframes / duration
    print("video %s: %dx%d %d frames %.3fs fps %.6f"
          % (os.path.basename(args.video), w, h, nframes, duration, fps))

    pf = PoseFinder(H, alpha=args.alpha, nadir=nadir, search=search,
                    frac_min=args.frac_min, resid_max=args.resid_max)
    fdet = open(args.dets, "r", encoding="utf-8")
    ffmpeg = common.find_ffmpeg(args.ffmpeg)
    out_mp4 = os.path.join(args.outdir, "overlay.mp4")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "%dx%d" % (w, h), "-framerate", "%.9f" % fps, "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=open(os.path.join(args.outdir, "ffmpeg.log"), "w"))
    fcsv = open(os.path.join(args.outdir, "pose.csv"), "w", newline="", encoding="utf-8")
    wr = csv.writer(fcsv)
    wr.writerow(["frame_idx", "timestamp", "x", "y", "yaw", "conf"])

    t0 = time.time()
    n = nok = nyaw = nref = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        line = fdet.readline()
        if not ok or not line:
            break
        rec = json.loads(line)
        dets = rec["d"]
        box = pick(dets, CHASSIS)
        ball = pick(dets, BALL)
        res = None
        platedet = None
        if box is not None:
            x1, y1, x2, y2 = box[2], box[3], box[4], box[5]
            m = 45
            roi = (int(x1) - m, int(y1) - m, int(x2) + m, int(y2) + m)
            d = pf.detect(frame, roi=roi)
            if d.get("ok"):
                if args.fuse and d.get("yaw_px") is None and d.get("axis") is not None:
                    u = np.array(d["axis"], float)
                    u /= (np.linalg.norm(u) or 1.0)
                    left = right = False
                    for bb in dets:
                        if bb[0] != BALL or bb[1] < 0.50:
                            continue
                        v = np.array([(bb[2] + bb[4]) / 2.0 - d["cx"],
                                      (bb[3] + bb[5]) / 2.0 - d["cy"]])
                        t = float(v @ u)
                        if abs(float(v[0] * u[1] - v[1] * u[0])) > 26.0:
                            continue
                        if 1.0 * d["r"] <= t <= 2.6 * d["r"]:
                            left = True
                        elif -2.6 * d["r"] <= t <= -1.0 * d["r"]:
                            right = True
                    if left and not right:
                        d["yaw_px"] = (float(u[0]), float(u[1]))
                    elif right and not left:
                        d["yaw_px"] = (float(-u[0]), float(-u[1]))
                r = pf.to_field(d)
                if r is not None and r[0] is not None:
                    platedet = d
                    res = r
                    nref += 1
            if res is None:
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0
                q = H @ np.array([u, v, 1.0])
                X, Y = q[0] / q[2], q[1] / q[2]
                s = 1.0 + args.alpha_box
                X = (nadir[0] + (X - nadir[0]) / s) / 1000.0
                Y = (nadir[1] + (Y - nadir[1]) / s) / 1000.0
                res = (X, Y, None, box[1] * 0.6)
        bres = None
        if ball is not None and ball[1] >= args.ball_conf:
            u = (ball[2] + ball[4]) / 2.0
            v = (ball[3] + ball[5]) / 2.0
            q = H @ np.array([u, v, 1.0])
            bres = (q[0] / q[2] / 1000.0, q[1] / q[2] / 1000.0, ball[1])

        vis = frame.copy()
        if box is not None:
            col = (60, 235, 60) if platedet else (0, 200, 255)
            cv2.rectangle(vis, (int(box[2]), int(box[3])), (int(box[4]), int(box[5])), col, 2)
        if platedet is not None:
            c = (int(platedet["cx"]), int(platedet["cy"]))
            R = int(platedet["r"])
            cv2.circle(vis, c, R, (60, 235, 60), 2)
            cv2.drawMarker(vis, c, (60, 235, 60), cv2.MARKER_CROSS, 18, 2)
            if platedet.get("yaw_px"):
                dx, dy = platedet["yaw_px"]
                cv2.arrowedLine(vis, c, (int(c[0] + dx * R * 2.6), int(c[1] + dy * R * 2.6)),
                                (30, 60, 255), 4, tipLength=0.22)
        if res is not None:
            x, y, yaw, conf = res
            txt = "x=%.3f m  y=%.3f m" % (x, y)
            txt += ("  yaw=%.1f deg" % yaw) if yaw is not None else "  yaw=n/a"
            if platedet is None:
                txt += "  [box]"
            cv2.putText(vis, txt, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(vis, txt, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (60, 235, 60) if platedet else (0, 200, 255), 2, cv2.LINE_AA)
            wr.writerow([idx, "%.6f" % pts[idx], "%.4f" % x, "%.4f" % y,
                         ("%.2f" % yaw) if yaw is not None else "NaN", "%.2f" % conf])
            nok += 1
            nyaw += 1 if yaw is not None else 0
        else:
            cv2.putText(vis, "no detection", (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            wr.writerow([idx, "%.6f" % pts[idx], "NaN", "NaN", "NaN", "NaN"])
        if bres:
            cv2.putText(vis, "ball %.3f %.3f" % (bres[0], bres[1]), (18, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        hud = "f%05d  t=%.2fs / %.1fs  yolo" % (idx, pts[idx], pts[nframes - 1])
        cv2.putText(vis, hud, (18, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, hud, (18, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
        proc.stdin.write(vis.tobytes())
        n += 1
        idx += 1
        if n % 500 == 0:
            el = time.time() - t0
            print("  %d/%d %.1f fps ok=%.1f%% yaw=%.1f%% refined=%.1f%%"
                  % (n, nframes, n / el, 100.0 * nok / n, 100.0 * nyaw / n, 100.0 * nref / n),
                  flush=True)
        if args.limit and n >= args.limit:
            break
    cap.release()
    proc.stdin.close()
    proc.wait()
    fcsv.close()
    stats = {"frames": n, "detected": nok, "with_yaw": nyaw, "refined": nref,
             "detected_frac": round(nok / max(1, n), 4),
             "yaw_frac": round(nyaw / max(1, n), 4),
             "refined_frac": round(nref / max(1, n), 4),
             "fps_video": fps, "width": w, "height": h, "elapsed_s": round(time.time() - t0, 1),
             "alpha": args.alpha, "alpha_box": args.alpha_box,
             "nadir_mm": list(nadir), "calib": os.path.abspath(calib), "mode": "yolo"}
    json.dump(stats, open(os.path.join(args.outdir, "stats.json"), "w"), indent=1)
    print("DONE", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
