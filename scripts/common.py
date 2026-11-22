"""Shared plumbing: ffmpeg discovery, calibration loading, mp4 frame timing."""
from __future__ import annotations

import json
import os
import shutil
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
DEFAULT_CALIB = os.path.join(ASSETS, "calib_kaiser_example.json")


# ---------------------------------------------------------------- ffmpeg ----
def find_ffmpeg(explicit: str | None = None) -> str:
    """Locate an ffmpeg with libx264: explicit flag > $POSE_FFMPEG > PATH > bundled."""
    if explicit:
        return explicit
    env = os.environ.get("POSE_FFMPEG")
    if env and os.path.exists(env):
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    for root, _dirs, files in os.walk(ASSETS):
        for f in files:
            if f.startswith("ffmpeg"):
                return os.path.join(root, f)
    raise SystemExit(
        "no ffmpeg found: pass --ffmpeg /path/to/ffmpeg, set POSE_FFMPEG, "
        "or put one in assets/ (needs libx264)")


# ------------------------------------------------------------ calibration ---
def resolve_calib(explicit: str | None = None) -> str:
    """Explicit path > $POSE_CALIB > assets/calib_kaiser_example.json."""
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit("calibration file not found: %s" % explicit)
        return explicit
    env = os.environ.get("POSE_CALIB")
    if env:
        if not os.path.exists(env):
            raise SystemExit("POSE_CALIB points at a missing file: %s" % env)
        return env
    if os.path.exists(DEFAULT_CALIB):
        return DEFAULT_CALIB
    raise SystemExit(
        "no calibration: run scripts/calibrate_field.py and pass --calib, "
        "or set POSE_CALIB")


def load_homography(path: str) -> np.ndarray:
    d = json.load(open(path))
    return np.array(d["H"], float)


# ------------------------------------------------------------- mp4 timing ---
def _boxes(data, start, end):
    p = start
    while p + 8 <= end:
        size, typ = struct.unpack(">I4s", data[p:p + 8])
        typ = typ.decode("latin1")
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", data[p + 8:p + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - p
        if size < hdr:
            break
        yield typ, p, p + size, p + hdr
        p += size


def _find(data, start, end, want):
    for typ, s, e, body in _boxes(data, start, end):
        if typ == want:
            yield s, e, body
        if typ in ("moov", "trak", "mdia", "minf", "stbl"):
            yield from _find(data, body, e, want)


def video_timing(path: str):
    """Per-frame presentation times straight from the container.

    Returns (timescale, duration_ticks, per-frame deltas, cumulative pts seconds).
    Phone / PTZ recordings are usually variable frame rate, so never reconstruct
    timestamps as frame_idx / fps.
    """
    data = open(path, "rb").read()
    mdhd = next(_find(data, 0, len(data), "mdhd"))
    body = mdhd[2]
    ver = data[body]
    if ver == 1:
        ts, dur = struct.unpack(">IQ", data[body + 20:body + 32])
    else:
        ts, dur = struct.unpack(">II", data[body + 12:body + 20])
    stts = next(_find(data, 0, len(data), "stts"))
    body = stts[2]
    n = struct.unpack(">I", data[body + 4:body + 8])[0]
    deltas = []
    p = body + 8
    for _ in range(n):
        cnt, delta = struct.unpack(">II", data[p:p + 8])
        p += 8
        deltas.extend([delta] * cnt)
    pts = np.concatenate([[0.0], np.cumsum(deltas)[:-1]]) / float(ts)
    return ts, dur, np.asarray(deltas), pts
