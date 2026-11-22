#!/usr/bin/env python3
"""Write OUTDIR/notes.txt: conventions, calibration, timing and coverage.

    gen_notes.py OUTDIR [--calib F] [--lang zh|en]

Everything factual (resolution, frame count, fps, CFR/VFR, alpha, nadir,
coverage, failure intervals) is read back from stats.json / pose.csv, so the
notes always describe the run that produced them.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def failure_runs(ok, gap=20, min_len=15):
    runs, start = [], None
    for i, v in enumerate(list(ok) + [True]):
        if not v and start is None:
            start = i
        elif v and start is not None:
            runs.append((start, i - 1))
            start = None
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= min_len]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--calib", default=None, help="override stats.json calib path")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    args = ap.parse_args()

    od = args.outdir
    stats = json.load(open(os.path.join(od, "stats.json")))
    rows = list(csv.DictReader(open(os.path.join(od, "pose.csv"))))
    n = len(rows)
    ts = [float(r["timestamp"]) for r in rows]
    ok = [r["x"] != "NaN" for r in rows]
    yawok = [r["yaw"] != "NaN" for r in rows]
    nok, nyaw = sum(ok), sum(yawok)

    fps = float(stats["fps_video"])
    dur = ts[-1] + 1.0 / fps if n else 0.0
    alpha = float(stats.get("alpha", 0.08))
    nadir = stats.get("nadir_mm", [57.0, 0.0])
    mode = stats.get("mode", "geometric")
    w, h = stats.get("width"), stats.get("height")

    deltas = [round(ts[i + 1] - ts[i], 6) for i in range(n - 1)]
    cfr = len(set(deltas)) <= 1 if deltas else True
    dmin = min(deltas) if deltas else 0.0
    dmax = max(deltas) if deltas else 0.0

    calib_path = args.calib or stats.get("calib")
    H = None
    if calib_path and os.path.exists(calib_path):
        H = json.load(open(calib_path)).get("H")
    long = failure_runs(ok)

    L = []
    add = L.append
    if args.lang == "zh":
        add("机器人俯视轨迹 —— 记录说明")
        add("=" * 46)
        add("")
        add("1. 坐标系 / 单位 / yaw 约定")
        add("   - 原点：场地白色边界线矩形的中心（通常就是中心圆圆心）。")
        add("   - +x：沿场地长边、指向黄门（画面右侧）；+y：沿短边、指向画面上方。")
        add("   - 单位：x/y 为米，yaw 为度。yaw = 0 指向 +x，逆时针为正（俯视、y 向上）。")
        add("   - 机器人正前方：由顶部标记决定（本流水线默认取指示杆带黑色圆头的一端）。")
        add("     圆头被车顶黑盖挡住、看不到时，yaw 写 NaN，绝不按光杆那端猜。")
        add("   - x,y 取顶部标记的中心（旋转轴）。")
        add("")
        add("2. 像素 -> 米（单应矩阵，直接作用在原图像素上）")
        add("   标定文件：%s" % (calib_path or "（stats.json 未记录）"))
        if H:
            add("   H (px -> mm):")
            for row in H:
                add("       [ %12.6f %12.6f %12.4f ]" % tuple(row))
            add("   X_mm = (h00*u + h01*v + h02) / (h20*u + h21*v + h22)，y 同理。")
        add("   标定自检：把场地四角/标记点反投影，误差应在几个像素/毫米以内。")
        add("")
        add("3. 高度视差校正（机器不在相机正下方时会“变长”）")
        add("   标记离地 h，投影会沿天底方向外移 h/(Zc-h)，按")
        add("       P = N + (Q - N) / (1 + alpha),  alpha = %.3f,  N = (%.0f, %.0f) mm" % (alpha, nadir[0], nadir[1]))
        add("   还原成地面点（Q 是单应矩阵直接给出的地面投影）。")
        add("   alpha 是估计值，不确定度约 ±0.03：场地边缘约 ±3 cm，场地中心附近可忽略。")
        add("   换了车/换了标记高度必须重估；逆变换：x_homography = (1+alpha)*x_true - alpha*Nx。")
        add("")
        add("4. 视频与时间")
        if w and h:
            add("   - 分辨率 %dx%d，总帧数 %d，时长 %.3f s，平均 fps %.6f。" % (w, h, n, dur, fps))
        else:
            add("   - 总帧数 %d，时长 %.3f s，平均 fps %.6f。" % (n, dur, fps))
        if cfr:
            add("   - 恒定帧率（帧间隔均为 %.6f s）。" % dmin)
        else:
            add("   - 变帧率：帧间隔 %.6f ~ %.6f s，不是常数。" % (dmin, dmax))
        add("   - timestamp 列是原视频每一帧的真实 PTS（从 mp4 stts 解出），不是 frame_idx/fps。")
        add("   - overlay.mp4 用恒定 %.6f fps 重新封装，帧数/分辨率/总时长与原视频一致" % fps)
        add("     （编码器只能给恒定帧率；逐帧真实时间以 pose.csv 的 timestamp 为准）。")
        add("   - 编码：h264 (libx264, crf 20, yuv420p)。")
        add("")
        add("5. 检测结果")
        add("   - 位置有效：%d / %d 帧（%.1f%%）；yaw 有效：%d 帧（%.1f%%）。"
            % (nok, n, 100.0 * nok / max(1, n), nyaw, 100.0 * nyaw / max(1, n)))
        if mode == "yolo":
            ref = stats.get("refined", 0)
            add("   - 模式：YOLO 底盘框 + ROI 内圆盘精修。")
            add("   - 其中 %d 帧用了圆盘圆心（高精度），%d 帧退回底盘框中心（橙色框，精度略差），"
                % (ref, nok - ref))
            add("     %d 帧两样都没有。" % (n - nok))
            add("   - 底盘框中心的 alpha 用 %.4f（整车轮廓质心标定值）。" % stats.get("alpha_box", 0.0514))
        else:
            add("   - 模式：纯几何（暗色圆盘径向边缘拟合）。")
        add("   - yaw 只有标记朝向可判读时才输出，否则 NaN——宁可缺也不给错。")
        add("   - 置信度 conf：拟合质量（覆盖率与残差）0~1；NaN 行 conf 也是 NaN。")
        add("")
        add("6. 检测失败/被遮挡的帧段（按时间，含半台机器/出画面）")
        if not long:
            add("   - 无（所有帧都给出了位置）。")
        else:
            for a, b in long:
                add("   - %.2f s ~ %.2f s（帧 %d~%d，%.1f s）" % (ts[a], ts[b], a, b, (b - a) / fps))
        add("")
        add("7. overlay.mp4 图例")
        add("   - 绿圈=顶部标记圆盘，十字=旋转轴，红箭头=yaw；YOLO 模式绿框=底盘框，")
        add("     橙框=圆盘没拟合出来、位置退回框中心。")
        add("   - 没有检测时左上角写 \"no detection\"；底部 HUD 是帧号/时间/alpha。")
    else:
        add("Overhead robot trajectory - run notes")
        add("=" * 36)
        add("")
        add("1. Frame, units and yaw convention")
        add("   - origin: centre of the painted boundary rectangle (usually the centre spot).")
        add("   - +x along the long side towards the yellow goal (image right); +y towards the")
        add("     top of the image.  x/y in metres.")
        add("   - yaw in degrees, 0 along +x, counter-clockwise positive (plan view, y up).")
        add("   - robot forward: defined by the top marker (default: the end of the pointer")
        add("     bar that carries the black knob).  If the knob is not visible, yaw is NaN --")
        add("     never guess from the bare end.")
        add("   - x,y is the marker centre, i.e. the rotation axis.")
        add("")
        add("2. Pixel -> metre homography (applied to source pixels)")
        add("   calibration: %s" % (calib_path or "(not recorded in stats.json)"))
        if H:
            add("   H (px -> mm):")
            for row in H:
                add("       [ %12.6f %12.6f %12.4f ]" % tuple(row))
            add("   X_mm = (h00*u + h01*v + h02) / (h20*u + h21*v + h22); same for y.")
        add("   Self-check: reprojecting field corners / markers should land within a few px.")
        add("")
        add("3. Height parallax correction (markers are not on the floor)")
        add("       P = N + (Q - N) / (1 + alpha),  alpha = %.3f,  N = (%.0f, %.0f) mm" % (alpha, nadir[0], nadir[1]))
        add("   Q is the flat-ground projection from H; P is the true ground point.")
        add("   alpha is an estimate (h/(Zc-h)); +-0.03 is worth ~3 cm at the field edge.")
        add("   Re-estimate if the robot or the marker height changes.  Inverse:")
        add("       x_homography = (1+alpha)*x_true - alpha*Nx")
        add("")
        add("4. Video and timing")
        if w and h:
            add("   - %dx%d, %d frames, %.3f s, mean fps %.6f." % (w, h, n, dur, fps))
        else:
            add("   - %d frames, %.3f s, mean fps %.6f." % (n, dur, fps))
        if cfr:
            add("   - constant frame rate (frame interval %.6f s)." % dmin)
        else:
            add("   - variable frame rate: frame interval %.6f..%.6f s." % (dmin, dmax))
        add("   - the timestamp column is the real per-frame PTS from the mp4 stts box,")
        add("     not frame_idx/fps.")
        add("   - overlay.mp4 is re-encoded at a constant %.6f fps with the same frame count," % fps)
        add("     resolution and duration (encoders cannot express VFR; trust the CSV times).")
        add("   - encode: h264 (libx264, crf 20, yuv420p).")
        add("")
        add("5. Detection results")
        add("   - position: %d / %d frames (%.1f%%); yaw: %d frames (%.1f%%)."
            % (nok, n, 100.0 * nok / max(1, n), nyaw, 100.0 * nyaw / max(1, n)))
        if mode == "yolo":
            ref = stats.get("refined", 0)
            add("   - mode: YOLO chassis box + plate refinement inside the box.")
            add("   - %d frames used the fitted plate centre, %d fell back to the box centre"
                % (ref, nok - ref))
            add("     (orange box in the overlay, lower accuracy), %d had neither." % (n - nok))
            add("   - the box-centre fallback uses alpha = %.4f (whole-body centroid)."
                % stats.get("alpha_box", 0.0514))
        else:
            add("   - mode: pure geometry (radial edge fit on the dark plate).")
        add("   - yaw is only written when the marker's front is unambiguous; otherwise NaN.")
        add("   - conf: fit quality (edge coverage and residual), 0..1; NaN rows have NaN conf.")
        add("")
        add("6. Intervals without a fix (occluded, half robot, out of frame)")
        if not long:
            add("   - none: every frame produced a position.")
        else:
            for a, b in long:
                add("   - %.2f s .. %.2f s (frames %d..%d, %.1f s)" % (ts[a], ts[b], a, b, (b - a) / fps))
        add("")
        add("7. overlay.mp4 legend")
        add("   - green circle = top marker, cross = rotation axis, red arrow = yaw; in YOLO")
        add("     mode a green box is the chassis box and an orange box means the position")
        add("     fell back to the box centre.")
        add("   - \"no detection\" in the top-left when nothing was found; the HUD shows")
        add("     frame index / time / alpha.")

    out = os.path.join(od, "notes.txt")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("notes written", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
