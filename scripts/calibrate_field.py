#!/usr/bin/env python3
"""Fit the pixel -> field-mm homography from the painted white boundary lines.

    calibrate_field.py --image BG.jpg --out calib.json [--half-long 1095 --half-short 790]
    calibrate_field.py --video REC.mp4 --frames 40 --out calib.json

The background frame must show the whole painted rectangle with the boundary
line visible near every image edge (the calibrator reads the first white run
from each edge).  The four fitted corner intersections are mapped to
(-HL,+HS), (+HL,+HS), (+HL,-HS), (-HL,-HS) mm, i.e. the field frame is centred
on the centre spot with +x along the long side and +y up.

Always sanity-check the printed px/mm scales: the four sides should agree
within a fraction of a percent on a perspective-free overhead camera.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np


def white_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 120)


def side_centres(mask, side, band=0.25, run=(3, 60)):
    """Centres of the first white run seen from one image edge, per column/row."""
    h, w = mask.shape
    pts = []
    lo, hi = run
    if side == "top":
        lim = int(h * band)
        for x in range(int(w * 0.12), int(w * 0.88)):
            col = mask[:lim, x]
            idx = np.nonzero(col)[0]
            if not len(idx):
                continue
            j = idx[0]
            k = j
            while k < lim and col[k]:
                k += 1
            if lo <= k - j <= hi:
                pts.append((x, (j + k - 1) / 2.0))
    elif side == "bottom":
        lim = int(h * (1 - band))
        for x in range(int(w * 0.12), int(w * 0.88)):
            col = mask[lim:, x]
            idx = np.nonzero(col)[0]
            if not len(idx):
                continue
            j = int(idx[-1])
            k = j
            while k >= 0 and col[k]:
                k -= 1
            if lo <= j - k <= hi:
                pts.append((x, lim + (j + k + 1) / 2.0))
    elif side == "left":
        lim = int(w * band)
        for y in range(int(h * 0.12), int(h * 0.88)):
            row = mask[y, :lim]
            idx = np.nonzero(row)[0]
            if not len(idx):
                continue
            j = idx[0]
            k = j
            while k < lim and row[k]:
                k += 1
            if lo <= k - j <= hi:
                pts.append(((j + k - 1) / 2.0, y))
    else:
        lim = int(w * (1 - band))
        for y in range(int(h * 0.12), int(h * 0.88)):
            row = mask[y, lim:]
            idx = np.nonzero(row)[0]
            if not len(idx):
                continue
            j = int(idx[-1])
            k = j
            while k >= 0 and row[k]:
                k -= 1
            if lo <= j - k <= hi:
                pts.append((lim + (j + k + 1) / 2.0, y))
    return np.asarray(pts, float)


def ransac_line(P, axis, thr=1.5, iters=1200, seed=1):
    """axis='y': y = a*x + b (horizontal line); axis='x': x = a*y + b."""
    if len(P) < 10:
        return None
    rng = np.random.RandomState(seed)
    best = (-1, None)
    for _ in range(iters):
        i, j = rng.randint(0, len(P), 2)
        if i == j:
            continue
        if axis == "y":
            if abs(P[j, 0] - P[i, 0]) < 60:
                continue
            a = (P[j, 1] - P[i, 1]) / (P[j, 0] - P[i, 0])
            b = P[i, 1] - a * P[i, 0]
            res = np.abs(P[:, 1] - (a * P[:, 0] + b))
        else:
            if abs(P[j, 1] - P[i, 1]) < 60:
                continue
            a = (P[j, 0] - P[i, 0]) / (P[j, 1] - P[i, 1])
            b = P[i, 0] - a * P[i, 1]
            res = np.abs(P[:, 0] - (a * P[:, 1] + b))
        n = int((res < thr).sum())
        if n > best[0]:
            best = (n, (a, b, res < thr))
    n, got = best
    if got is None or n < 10:
        return None
    a, b, inl = got
    Q = P[inl]
    if axis == "y":
        k, c = np.polyfit(Q[:, 0], Q[:, 1], 1)
        rms = float(np.sqrt(np.mean((Q[:, 1] - (k * Q[:, 0] + c)) ** 2)))
    else:
        k, c = np.polyfit(Q[:, 1], Q[:, 0], 1)
        rms = float(np.sqrt(np.mean((Q[:, 0] - (k * Q[:, 1] + c)) ** 2)))
    return dict(a=float(k), b=float(c), inliers=int(n), total=int(len(P)), rms=rms,
                pts=Q, axis=axis)


def intersect(Lh, Lv):
    """Horizontal line y=a*x+b meets vertical line x=a*y+b."""
    ay, by = Lh["a"], Lh["b"]
    ax, bx = Lv["a"], Lv["b"]
    y = (ay * bx + by) / (1.0 - ay * ax)
    x = ax * y + bx
    return (x, y)


def background(args):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit("cannot read image: %s" % args.image)
        return img
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("cannot open video: %s" % args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max(1, args.frames))
    acc = None
    n = 0
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        acc = g if acc is None else np.minimum(acc, g)   # min = brightest floor, keeps the paint
        n += 1
        if n >= args.frames:
            break
    if acc is None:
        raise SystemExit("could not read frames from %s" % args.video)
    return cv2.cvtColor(acc.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="background frame (no robot on the mat)")
    src.add_argument("--video", help="video to build a background from")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--half-long", type=float, default=1095.0, help="mm, centre to long edge")
    ap.add_argument("--half-short", type=float, default=790.0, help="mm, centre to short edge")
    ap.add_argument("--band", type=float, default=0.25, help="search band from each edge, fraction")
    ap.add_argument("--corners", default=None,
                    help="skip detection: 'TLx,TLy;TRx,TRy;BRx,BRy;BLx,BLy' in source pixels")
    ap.add_argument("--preview", default=None, help="write a PNG with the fitted quad drawn")
    ap.add_argument("--out", default="calib_field.json")
    args = ap.parse_args()

    img = background(args)
    h, w = img.shape[:2]
    mask = white_mask(img)
    print("background %dx%d, white pixels %.2f%%" % (w, h, 100.0 * mask.mean()))

    if args.corners:
        pts = [[float(v) for v in p.split(",")] for p in args.corners.split(";") if p.strip()]
        if len(pts) != 4:
            raise SystemExit("--corners needs 4 'x,y' pairs separated by ';'")
        cor = np.array(pts, float)
        print("using corners given on the command line")
        return finish(img, cor, args, h, w)

    found = {}
    for side, axis in (("top", "y"), ("bottom", "y"), ("left", "x"), ("right", "x")):
        P = side_centres(mask, side, band=args.band)
        if len(P) < 10:
            print("  %-6s only %d candidate points - widen --band" % (side, len(P)))
            return 2
        L = ransac_line(P, axis)
        if L is None:
            print("  %-6s line fit failed (%d points)" % (side, len(P)))
            return 2
        found[side] = L
        print("  %-6s %5d pts, %4d inliers, rms %.2f px" % (side, L["total"], L["inliers"], L["rms"]))

    cor = np.array([intersect(found["top"], found["left"]),
                    intersect(found["top"], found["right"]),
                    intersect(found["bottom"], found["right"]),
                    intersect(found["bottom"], found["left"])], float)
    return finish(img, cor, args, h, w)


def finish(img, cor, args, h, w):
    HL, HS = args.half_long, args.half_short
    dst = np.array([[-HL, HS], [HL, HS], [HL, -HS], [-HL, -HS]], float)
    Hpx = cv2.getPerspectiveTransform(cor.astype(np.float32), dst.astype(np.float32))

    print("corners TL TR BR BL:\n", np.round(cor, 2))
    sc = [np.linalg.norm(cor[1] - cor[0]) / (2 * HL), np.linalg.norm(cor[2] - cor[3]) / (2 * HL),
          np.linalg.norm(cor[3] - cor[0]) / (2 * HS), np.linalg.norm(cor[2] - cor[1]) / (2 * HS)]
    print("scale px/mm: top %.4f bottom %.4f left %.4f right %.4f" % tuple(sc))
    spread = (max(sc) - min(sc)) / float(np.mean(sc))
    if spread > 0.03:
        print("WARNING: the four sides disagree by %.1f%% in scale - the quad probably does" % (100 * spread))
        print("         not follow the painted line.  Check the preview and pass --corners.")
    else:
        print("scale spread %.2f%% (ok)" % (100 * spread))
    q = Hpx @ np.array([w / 2.0, h / 2.0, 1.0])
    print("image centre -> field mm %s" % np.round((q / q[2])[:2], 1))
    back = cv2.perspectiveTransform(cor.reshape(-1, 1, 2), Hpx).reshape(-1, 2)
    print("corner reprojection error (mm) max %.2f" % float(np.abs(back - dst).max()))

    out = {"corners": cor.tolist(), "H": Hpx.tolist(), "H_inv": np.linalg.inv(Hpx).tolist(),
           "half_long_mm": HL, "half_short_mm": HS, "image_size": [w, h],
           "source": args.image or args.video}
    prev = args.preview or (os.path.splitext(args.out)[0] + ".preview.png")
    vis = img.copy()
    cv2.polylines(vis, [cor.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3)
    for name, p in zip(("TL", "TR", "BR", "BL"), cor):
        cv2.circle(vis, (int(p[0]), int(p[1])), 8, (0, 0, 255), -1)
        cv2.putText(vis, "%s (%d,%d)" % (name, p[0], p[1]), (int(p[0]) + 12, int(p[1]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(prev, cv2.resize(vis, None, fx=0.5, fy=0.5))
    print("preview:", prev, "- check that the red quad follows the painted line")

    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
