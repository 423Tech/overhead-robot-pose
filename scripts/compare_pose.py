"""Compare two pose.csv files frame by frame."""
import csv, math, sys
import numpy as np


def load(p):
    d = {}
    for r in csv.DictReader(open(p)):
        i = int(r["frame_idx"])
        d[i] = (None if r["x"] == "NaN" else float(r["x"]),
                None if r["y"] == "NaN" else float(r["y"]),
                None if r["yaw"] == "NaN" else float(r["yaw"]),
                None if r["conf"] == "NaN" else float(r["conf"]))
    return d


def ang_diff(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def main():
    a = load(sys.argv[1]); b = load(sys.argv[2])
    keys = sorted(set(a) & set(b))
    na = sum(1 for v in a.values() if v[0] is not None)
    nb = sum(1 for v in b.values() if v[0] is not None)
    print("%-28s frames=%d pos=%d (%.1f%%) yaw=%d (%.1f%%)" % (sys.argv[1].split("/")[-2], len(a), na, 100.0*na/len(a),
          sum(1 for v in a.values() if v[2] is not None), 100.0*sum(1 for v in a.values() if v[2] is not None)/len(a)))
    print("%-28s frames=%d pos=%d (%.1f%%) yaw=%d (%.1f%%)" % (sys.argv[2].split("/")[-2], len(b), nb, 100.0*nb/len(b),
          sum(1 for v in b.values() if v[2] is not None), 100.0*sum(1 for v in b.values() if v[2] is not None)/len(b)))
    both = [k for k in keys if a[k][0] is not None and b[k][0] is not None]
    if both:
        d = np.array([math.hypot(a[k][0]-b[k][0], a[k][1]-b[k][1]) for k in both])
        print("both have position: %d;  |delta| median %.3f m  p90 %.3f  p99 %.3f  max %.3f" % (
            len(both), np.median(d), np.percentile(d, 90), np.percentile(d, 99), d.max()))
        big = (d > 0.15).sum()
        print("   frames with >15cm disagreement: %d (%.1f%%)" % (big, 100.0*big/len(both)))
    byy = [k for k in keys if a[k][2] is not None and b[k][2] is not None]
    if byy:
        dd = np.array([ang_diff(a[k][2], b[k][2]) for k in byy])
        print("both have yaw: %d;  median diff %.1f deg  p90 %.1f  |diff|>30deg: %d (%.1f%%)" % (
            len(byy), np.median(dd), np.percentile(dd, 90), (dd > 30).sum(), 100.0*(dd > 30).sum()/len(byy)))


if __name__ == "__main__":
    main()
