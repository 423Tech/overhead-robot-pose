"""Robot pose from the Kaiser overhead field camera (skemma + heading marker).

Geometry (see notes.txt):
  * px -> field mm via the painted boundary-line homography in calib_field.json;
  * field frame: origin at the centre spot, +x towards the yellow goal (image
    right), +y towards the top of the image, yaw = 0 along +x, CCW positive;
  * the top marker (lidar plate + pointer rod) sits ALPHA * Zc above the mat,
    so the flat-ground projection is pulled back towards the nadir by
    P = N + (Q - N) / (1 + ALPHA).
"""
from __future__ import annotations
import json
import math
import numpy as np
import cv2

ALPHA = 0.08              # marker-plane parallax ratio h / (Zc - h)
NADIR_MM = (57.0, 0.0)    # camera nadir in the centred field frame
IMG_SEARCH = (170, 0, 1750, 1080)   # mat region in pixels

DISC_AREA = (2200, 32000)
DISC_FILL_MIN = 0.42
DISC_ASPECT = (0.55, 1.8)
DISC_R = (42.0, 100.0)
DISC_FRAC_MIN = 0.22
DISC_RESID_MAX = 4.0
BAR_AREA_MIN = 110
BAR_LEN = (60.0, 340.0)
BAR_ASPECT = 3.2
BAR_LO = np.array([108, 55, 22], np.uint8)
BAR_HI = np.array([188, 235, 255], np.uint8)


def load_homography(path):
    d = json.load(open(path))
    return np.array(d["H"], float)


def height_correct(x_mm, y_mm, alpha=ALPHA, nadir=NADIR_MM):
    """Undo the marker-height parallax: ground = nadir + (projected - nadir)/(1+alpha)."""
    s = 1.0 + alpha
    return ((x_mm + alpha * nadir[0]) / s, (y_mm + alpha * nadir[1]) / s)


def px_to_field(H, u, v):
    q = H @ np.array([u, v, 1.0])
    if abs(q[2]) < 1e-12:
        return None
    return (q[0] / q[2], q[1] / q[2])


class PoseFinder:
    """Finds the robot's rotation axis and heading in one frame.

    Every constant below has a module-level default tuned on the Kaiser lab
    camera; override them per rig through the constructor / CLI flags.
    """

    def __init__(self, H, alpha=ALPHA, nadir=NADIR_MM, search=IMG_SEARCH,
                 frac_min=DISC_FRAC_MIN, resid_max=DISC_RESID_MAX):
        self.H = np.asarray(H, float)
        self.alpha = float(alpha)
        self.nadir = (float(nadir[0]), float(nadir[1]))
        self.search = tuple(search)
        self.frac_min = float(frac_min)
        self.resid_max = float(resid_max)
        self.H_inv = np.linalg.inv(self.H)

    # -- circle fitted to the dark lidar plate -------------------------------
    def _fit_circle(self, gray, c0, rmin=40.0, rmax=105.0, nth=64, nrad=110):
        H, W = gray.shape
        r = np.linspace(rmin, rmax, nrad)
        a = np.linspace(0, 2 * np.pi, nth, endpoint=False)
        ca, sa = np.cos(a)[:, None], np.sin(a)[:, None]
        xs = c0[0] + ca * r[None, :]
        ys = c0[1] + sa * r[None, :]
        valid = (xs >= 1) & (ys >= 1) & (xs < W - 2) & (ys < H - 2)
        if valid.sum() < 400:
            return None
        xi = np.clip(xs.astype(np.int32), 0, W - 1)
        yi = np.clip(ys.astype(np.int32), 0, H - 1)
        prof = gray[yi, xi].astype(np.float32)
        d = np.gradient(prof, axis=1)
        d[~valid] = -1e9
        idx = np.argmax(d, axis=1)
        best = d[np.arange(nth), idx]
        ok = best > 8.0
        if ok.sum() < 10:
            return None
        pts = np.stack([xs[np.arange(nth), idx][ok], ys[np.arange(nth), idx][ok]], 1)

        def fit(P):
            A = np.column_stack([2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))])
            b = (P ** 2).sum(1)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy, c = sol
            return cx, cy, math.sqrt(max(c + cx * cx + cy * cy, 0.0))

        cx, cy, rr = fit(pts)
        for _ in range(3):
            dd = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - rr)
            keep = dd < max(2.5, float(np.percentile(dd, 70)))
            if keep.sum() < 8:
                break
            pts = pts[keep]
            cx, cy, rr = fit(pts)
        dd = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - rr)
        return dict(cx=float(cx), cy=float(cy), r=float(rr), n=int(len(pts)),
                    frac=float(len(pts)) / nth, resid=float(np.median(dd)))

    def _dark_candidates(self, gray, roi=None):
        x0, y0, x1, y1 = roi if roi else self.search
        h, w = gray.shape[:2]
        x0 = max(0, min(int(x0), w)); x1 = max(0, min(int(x1), w))
        y0 = max(0, min(int(y0), h)); y1 = max(0, min(int(y1), h))
        dark = (gray[y0:y1, x0:x1] < 92).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, ct = cv2.connectedComponentsWithStats(dark, 8)
        out = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if not (DISC_AREA[0] <= a <= DISC_AREA[1]):
                continue
            fill = a / float(w * h)
            if fill < DISC_FILL_MIN:
                continue
            if not (DISC_ASPECT[0] <= w / float(max(h, 1)) <= DISC_ASPECT[1]):
                continue
            out.append((float(ct[i][0] + x0), float(ct[i][1] + y0), int(a), float(fill)))
        return out

    def _heading(self, hsv, c, R):
        """Direction of the lavender pointer rod around the plate.

        Colour alone is not enough: the rod swings from bright lavender to almost
        black with the lighting, the black plate under it has purple reflections,
        and the rod's cast shadow is a second, purple-ish streak pointing another
        way.  For every direction we therefore walk outward from just outside the
        plate and measure the longest continuous rod-coloured run; the real rod is
        the streak that starts at the plate rim and reaches furthest.  If the
        opposite direction gives a comparable run (shadow / second robot) the
        answer is ambiguous and we return None instead of guessing.
        """
        h = hsv[:, :, 0].astype(np.int16)
        sat = hsv[:, :, 1].astype(np.int16)
        val = hsv[:, :, 2].astype(np.int16)
        m = ((h >= 106) & (h <= 190) & (sat >= 42) & (val >= 18))
        Hh, Ww = m.shape
        nth = 96
        ang = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False)
        r0 = 1.10 * R
        rs = np.arange(r0, r0 + 250.0, 1.0)
        xs = c[0] + np.cos(ang)[:, None] * rs[None, :]
        ys = c[1] + np.sin(ang)[:, None] * rs[None, :]
        ok = (xs >= 0) & (ys >= 0) & (xs < Ww - 1) & (ys < Hh - 1)
        xi = np.clip(xs.astype(np.int32), 0, Ww - 1)
        yi = np.clip(ys.astype(np.int32), 0, Hh - 1)
        mm = m[yi, xi] & ok
        runs = np.zeros(nth)
        for k in range(nth):
            row = mm[k]
            idx = np.nonzero(row)[0]
            if len(idx) == 0:
                continue
            i0 = int(idx[0])
            if rs[i0] > r0 + 32.0:
                continue
            j = i0
            gap = 0
            length = 0.0
            while j < len(row):
                if row[j]:
                    gap = 0
                    if rs[j] - rs[i0] + 1.0 > length:
                        length = rs[j] - rs[i0] + 1.0
                else:
                    gap += 1
                    if gap > 5:
                        break
                j += 1
            runs[k] = length
        k = int(np.argmax(runs))
        best = float(runs[k])
        back = np.abs(np.angle(np.exp(1j * (ang - ang[k] - np.pi)))) <= np.deg2rad(40)
        opp = float(runs[back].max()) if back.any() else 0.0
        if best < 50.0 or best < opp + 10.0:
            return None, best, opp
        d = np.abs(np.angle(np.exp(1j * (ang - ang[k])))) <= np.deg2rad(14)
        sec = mm[d]
        px = xs[d][sec]
        py = ys[d][sec]
        if len(px) < 30:
            return None, best, opp
        P = np.stack([px, py], 1).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(P, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        if (x0 - c[0]) * vx + (y0 - c[1]) * vy < 0:
            vx, vy = -vx, -vy
        n = math.hypot(vx, vy) or 1.0
        return (vx / n, vy / n), best, opp

    @staticmethod
    def _ball_at_end(gray, c, u, R):
        """Is the black front sphere at this end of the pointer bar?

        The pointer is a bar through the rotation axis: one end carries a black
        sphere, the other is bare.  Past the plate rim the sphere is a compact
        dark blob with mat on *both* sides, which the dark robot body (one long
        dark stretch) does not look like.
        """
        Hh, Ww = gray.shape
        best = 0
        for off in (-3.0, 0.0, 3.0):
            ts = np.arange(1.15 * R, 205.0, 1.5)
            vals = []
            for t in ts:
                x = c[0] + u[0] * t - u[1] * off
                y = c[1] + u[1] * t + u[0] * off
                if x < 6 or y < 6 or x >= Ww - 6 or y >= Hh - 6:
                    vals.append(255.0)
                else:
                    vals.append(float(gray[int(y), int(x)]))
            dark = np.asarray(vals) < 105.0
            i = 0
            n = len(dark)
            while i < n:
                if dark[i]:
                    j = i
                    while j < n and dark[j]:
                        j += 1
                    before = i          # bright samples before this run
                    after = n - j       # bright samples after it
                    if (j - i) >= 6 and before >= 4 and after >= 3:
                        best = max(best, j - i)
                    i = j
                else:
                    i += 1
        return best >= 6

    def detect(self, frame, roi=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cands = self._dark_candidates(gray, roi)
        if not cands:
            return {"ok": False, "why": "no-plate"}
        fits = []
        for cx, cy, a, fill in cands:
            f = self._fit_circle(gray, (cx, cy))
            if f is None:
                continue
            if not (DISC_R[0] <= f["r"] <= DISC_R[1]):
                continue
            if math.hypot(f["cx"] - cx, f["cy"] - cy) > 42:
                continue
            if f["frac"] < self.frac_min or f["resid"] > self.resid_max:
                continue
            fits.append(f)
        if not fits:
            return {"ok": False, "why": "no-plate"}
        # prefer a well covered plate; if two plates compete, keep the better one
        fits.sort(key=lambda f: (-f["frac"], f["resid"]))
        disc = fits[0]
        c = (disc["cx"], disc["cy"])
        R = disc["r"]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yaw_px, hb, hs = self._heading(hsv, c, R)
        axis = None if yaw_px is None else (float(yaw_px[0]), float(yaw_px[1]))
        if yaw_px is not None:
            u = np.array(yaw_px, float)
            u /= np.linalg.norm(u) or 1.0
            # the sphere sits at one end of the bar; look for it on both sides
            fwd = self._ball_at_end(gray, np.array(c), u, R)
            bwd = self._ball_at_end(gray, np.array(c), -u, R)
            if fwd and not bwd:
                yaw_px = (float(u[0]), float(u[1]))
            elif bwd and not fwd:
                yaw_px = (float(-u[0]), float(-u[1]))
            else:
                # sphere not visible (merged with the black top cover) or visible
                # on both sides: do not guess.
                yaw_px = None
        conf = min(1.0, max(0.0, (disc["frac"] - 0.12) / 0.22)) * max(0.0, 1.0 - disc["resid"] / 5.0)
        return {"ok": True, "cx": c[0], "cy": c[1], "r": R, "yaw_px": yaw_px, "axis": axis,
                "conf": round(conf, 3), "frac": disc["frac"], "resid": disc["resid"],
                "why": "", "np": len(fits), "head": (hb, hs)}

    def to_field(self, det):
        """Pixel detection -> (x_m, y_m, yaw_deg, conf)."""
        q = px_to_field(self.H, det["cx"], det["cy"])
        if q is None:
            return None
        x, y = height_correct(q[0], q[1], self.alpha, self.nadir)
        yaw = None
        if det.get("yaw_px"):
            dx, dy = det["yaw_px"]
            p0 = px_to_field(self.H, det["cx"], det["cy"])
            p1 = px_to_field(self.H, det["cx"] + dx * 30.0, det["cy"] + dy * 30.0)
            if p0 and p1:
                yaw = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 360.0
        return x / 1000.0, y / 1000.0, yaw, det.get("conf", 0.0)
