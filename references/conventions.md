# Conventions, units and timebase

Everything here is what the reference implementation actually does. If you change
one of these, change the notes template too - a dataset whose documentation
disagrees with its numbers is worse than no dataset.

## Field frame

| property | value |
|---|---|
| origin | centre of the painted boundary rectangle (normally the centre spot) |
| +x | along the long side, towards the yellow goal (image right) |
| +y | along the short side, towards the top of the image |
| units | metres for x/y, degrees for yaw |
| yaw = 0 | along +x |
| sign | counter-clockwise positive in a plan view with +y up |

The homography maps the four painted corners to `(-HL,+HS)`, `(+HL,+HS)`,
`(+HL,-HS)`, `(-HL,-HS)` mm; `HL`/`HS` are half the *painted line centre*
dimensions (reference field: 1095 / 790 mm). Because the mapping is built on the
line centreline, the field frame is the geometric centre of the paint, not the
centre of the playable inner area - state which one you used.

## What "robot position" means

`x,y` is the centre of the robot's **top marker** (typically the lidar plate on
top of the chassis), which is the rotation axis of a differential drive robot. It
is *not* the centre of the bounding box of the whole robot, and not the centre of
the wheels.

Why it matters: a bbox drawn around the whole robot includes the pointer bar, the
shadow and the body silhouette, so its centre drifts by 5-15 cm depending on which
way the robot faces. A circle fitted to the top plate is stable to a few
millimetres. When a detector's box centre is used as a fallback, say so and mark
the rows (the reference marks them `conf * 0.6` and draws an orange box).

## Heading

`yaw` is where the robot's **front** points. The marker defines the front, not the
geometry of the fit:

- reference robot: lavender pointer bar through the axis, black knob at one end;
  the knob end is the front (`field ops` convention, confirmed by the operator);
- if the knob is not visible, write `NaN` - the bare end is *not* the back;
- if both ends look equally plausible, write `NaN`.

A 180-degree error is the single most common silent failure in this kind of work,
because averaged metrics do not notice it. Validate against one frame the user
confirms by eye, and prefer NaN when unsure.

## Height parallax

The marker sits `h` above the mat; the camera sits at height `Zc` with its nadir
at `N` in field coordinates (usually the field centre). A flat-ground homography
puts the marker's *projection* at

```
Q = N + (r_true + r_true * h/(Zc-h)) ...   i.e.  Q = N + r_true * Zc/(Zc-h)
```

so the true ground point is recovered with

```
P   = N + (Q - N) / (1 + alpha),      alpha = h / (Zc - h)
```

and the inverse (raw homography value, if someone wants it) is

```
Q   = N + (1 + alpha) * (P - N)
```

`alpha` is a *radial* scale about the nadir: it never changes the heading, it only
moves points in or out. On a 2.2 m field seen from ~3 m, a marker 0.2 m up gives
`alpha ~ 0.07`, which is ~7 cm of radial bias at 1 m from the centre - far larger
than the circle-fit noise, so do not skip it and do not guess it silently.

Two independent values are useful:

- `alpha_marker` for the top-plate centre (reference: 0.08, uncertain +-0.03);
- `alpha_box` for a whole-body bbox centre (field ops calibrated 0.0514 against
  the actual robot position).

Document which one produced each row when the pipeline mixes them.

## Timebase

- `timestamp` is the per-frame **presentation time** from the container
  (`stts` sample durations / mdhd timescale), cumulative from zero. Never
  reconstruct it as `frame_idx / fps`.
- Detect VFR honestly: if consecutive deltas differ, say "variable frame rate" and
  print the range (reference: 95-104 ticks at timescale 3000, i.e. 31.7-34.7 ms,
  with occasional 16-19 and 62-72 tick frames).
- Encoders cannot emit VFR from a raw pipe, so the overlay is written at the *mean*
  fps = `n_frames / (pts[-1] + last_delta)`. Frame count and duration still match
  the source; only the per-frame cadence is regularised. State this.
- Verify: `ffmpeg -i REC.mp4 -vf showinfo -frames:v 6 -f null -` prints the first
  PTS values; they must match your CSV.
