#!/usr/bin/env python3
"""Per-frame robot pose + annotated overlay for one recording.

    run_pose.py VIDEO OUTDIR [--calib F] [--alpha 0.08] [--nadir 57,0]
                [--start N] [--limit N] [--no-overlay]

Writes OUTDIR/pose.csv, OUTDIR/overlay.mp4 (unless --no-overlay) and
OUTDIR/stats.json.  One CSV row per source frame; undetectable fields are NaN.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
from pose_core import PoseFinder, ALPHA, NADIR_MM, IMG_SEARCH, DISC_FRAC_MIN, DISC_RESID_MAX

HEADER = ["frame_idx", "timestamp", "x", "y", "yaw", "conf"]


def overlay(frame, det, res, fidx, tsec, total, alpha):
    vis = frame.copy()
    h, w = vis.shape[:2]
    if det.get("ok"):
        c = (int(det["cx"]), int(det["cy"]))
        r = int(det["r"])
        cv2.circle(vis, c, r, (60, 235, 60), 2)
        cv2.drawMarker(vis, c, (60, 235, 60), cv2.MARKER_CROSS, 18, 2)
        if det.get("yaw_px"):
            dx, dy = det["yaw_px"]
            tip = (int(c[0] + dx * r * 2.6), int(c[1] + dy * r * 2.6))
            cv2.arrowedLine(vis, c, tip, (30, 60, 255), 4, tipLength=0.22)
    if det.get("ok") and res:
        x, y, yaw, conf = res
        txt = "x=%.3f m  y=%.3f m" % (x, y)
        txt += "  yaw=%.1f deg" % yaw if yaw is not None else "  yaw=n/a"
        col = (60, 235, 60)
    else:
        txt = "no detection"
        col = (0, 0, 255)
    cv2.putText(vis, txt, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(vis, txt, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
    hud = "f%05d  t=%.2fs / %.1fs  alpha=%.2f" % (fidx, tsec, total, alpha)
    cv2.putText(vis, hud, (18, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, hud, (18, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
    return vis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("outdir")
    ap.add_argument("--calib", default=None, help="pixel->field-mm homography json")
    ap.add_argument("--alpha", type=float, default=ALPHA, help="marker parallax h/(Zc-h)")
    ap.add_argument("--nadir", default="%g,%g" % NADIR_MM, help="camera nadir in field mm: x,y")
    ap.add_argument("--search", default="%d,%d,%d,%d" % IMG_SEARCH, help="mat bbox in px")
    ap.add_argument("--frac-min", type=float, default=DISC_FRAC_MIN)
    ap.add_argument("--resid-max", type=float, default=DISC_RESID_MAX)
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-overlay", action="store_true")
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
    print("video %s: %dx%d  %d frames  duration %.3fs  fps %.6f"
          % (os.path.basename(args.video), w, h, nframes, duration, fps))

    pf = PoseFinder(H, alpha=args.alpha, nadir=nadir, search=search,
                    frac_min=args.frac_min, resid_max=args.resid_max)

    proc = None
    if not args.no_overlay:
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
    wr.writerow(HEADER)
    t0 = time.time()
    n = nok = nyaw = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx < args.start:
            idx += 1
            continue
        if args.limit and idx >= args.start + args.limit:
            break
        det = pf.detect(frame)
        res = None
        if det.get("ok"):
            res = pf.to_field(det)
            if res is None or res[0] is None:
                det = {"ok": False}
                res = None
        if res is not None:
            x, y, yaw, conf = res
            nok += 1
            nyaw += 1 if yaw is not None else 0
            wr.writerow([idx, "%.6f" % pts[idx], "%.4f" % x, "%.4f" % y,
                         ("%.2f" % yaw) if yaw is not None else "NaN", "%.2f" % conf])
        else:
            wr.writerow([idx, "%.6f" % pts[idx], "NaN", "NaN", "NaN", "NaN"])
        if proc is not None:
            vis = overlay(frame, det if res is not None else {"ok": False}, res,
                          idx, pts[idx], pts[nframes - 1], args.alpha)
            proc.stdin.write(vis.tobytes())
        n += 1
        idx += 1
        if n % 500 == 0:
            el = time.time() - t0
            print("  %d/%d  %.1f fps  ok=%.1f%% yaw=%.1f%%"
                  % (n, nframes - args.start, n / el, 100.0 * nok / n, 100.0 * nyaw / n),
                  flush=True)
    cap.release()
    if proc is not None:
        proc.stdin.close()
        proc.wait()
    fcsv.close()
    stats = {
        "frames": n, "detected": nok, "with_yaw": nyaw,
        "detected_frac": round(nok / max(1, n), 4),
        "yaw_frac": round(nyaw / max(1, n), 4),
        "fps_video": fps, "width": w, "height": h, "elapsed_s": round(time.time() - t0, 1),
        "alpha": args.alpha, "nadir_mm": list(nadir), "search": list(search),
        "calib": os.path.abspath(calib), "mode": "geometric",
    }
    json.dump(stats, open(os.path.join(args.outdir, "stats.json"), "w"), indent=1)
    print("DONE", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
