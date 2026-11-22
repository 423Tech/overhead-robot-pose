# Field calibration

## What it produces

`calib_field.json`

```json
{
  "corners": [[231.6, 41.9], [1627.2, 24.1], [1622.1, 1042.2], [235.9, 1044.6]],
  "H": [[...], [...], [...]],
  "H_inv": [[...], [...], [...]],
  "half_long_mm": 1095.0,
  "half_short_mm": 790.0,
  "image_size": [1920, 1080]
}
```

`H` maps source pixels `(u,v,1)` to field **millimetres**:

```
X_mm = (h00*u + h01*v + h02) / (h20*u + h21*v + h22)
```

Divide by 1000 for metres. `H_inv` maps field mm back to pixels, which is handy
for drawing.

## How the reference calibration was made

1. Build a background: median (or per-pixel minimum) over ~20-40 frames spread
   across the recording, so the robot, its shadow and the ball disappear.
2. White mask in HSV: low saturation, high value. The painted line is the only
   long thin white structure inside the mat.
3. For each of the four image sides, scan the **central 70%** of the side (corners
   are full of goals, walls and labels) and take the centre of the first thin
   white run seen from the image edge. That gives a cloud of centreline points for
   that side.
4. RANSAC a line through each cloud (threshold ~1.5 px), then refine with a least
   squares fit on the inliers.
5. Intersect the four lines to get four corners in image coordinates.
6. `cv2.getPerspectiveTransform(corners, [[-HL,+HS],[+HL,+HS],[+HL,-HS],[-HL,-HS]])`.

Reference result: per-side rms 0.43-0.68 px, corners reproducible to <1 px,
scale ~0.635 px/mm (i.e. ~1.57 mm per pixel).

## Validation that actually catches mistakes

- **Corner preview**: `calibrate_field.py` writes `<out>.preview.png` with the quad
  drawn. Look at it. A quad that is 80 px too wide still produces a beautiful
  homography.
- **Scale spread**: the four side lengths in px/mm must agree. The reference rig
  shows 1.8% spread (a little perspective); anything above ~3% means the quad is
  not on the paint.
- **Known marks**: if the field has measured dots, corner arcs or a centre circle,
  reproject them and compare. The reference field has four dots at (+-640, +-380)
  mm; the calibration reproduces them within 3 mm. This is the strongest check and
  it costs one short script.
- **Nadir**: `H @ (w/2, h/2)` should land near the field centre if the camera looks
  at the middle of the field. On the reference rig it gives (57, 0) mm, which is
  the parallax nadir used by `--nadir`.

## When auto-detection fails

Symptoms: a side line found 50-100 px outside the paint (a wall, a desk edge, a
goal board), a quad that is not a rectangle, scale spread > 3%.

The painted boundary is not always the first white thing from the image edge -
goal boards, cable ducts and wall trims are also bright. Do not fight it: read the
corners off the preview (or from an image editor) and pass them explicitly:

```bash
python scripts/calibrate_field.py --image bg.jpg \
  --corners "231.6,41.9;1627.2,24.1;1622.1,1042.2;235.9,1044.6" \
  --out calib.json
```

Calibration is a one-time, human-verified step per camera. Treat it as such.
