"""Streaming YOLO inference over a video -> one JSONL line per source frame."""
import argparse, json, time

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    print("classes:", model.names, flush=True)
    fh = open(args.out, "w", encoding="utf-8", buffering=1 << 20)
    counts = {v: 0 for v in model.names.values()}
    t0 = time.time()
    n = 0
    gen = model.predict(source=args.video, stream=True, imgsz=args.imgsz, conf=args.conf,
                        iou=args.iou, verbose=False, device=args.device, half=True, vid_stride=1)
    for i, r in enumerate(gen):
        dets = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for bb, cc, kk in zip(xyxy, conf, cls):
                dets.append([int(kk), round(float(cc), 3),
                             round(float(bb[0]), 1), round(float(bb[1]), 1),
                             round(float(bb[2]), 1), round(float(bb[3]), 1)])
                counts[model.names[int(kk)]] += 1
        fh.write(json.dumps({"f": i, "d": dets}) + "\n")
        n += 1
        if n % 1000 == 0:
            el = time.time() - t0
            print("  %d frames  %.1f fps  %.0fs" % (n, n / el, el), flush=True)
        if args.limit and n >= args.limit:
            break
    fh.close()
    el = time.time() - t0
    print("DONE frames=%d %.1fs %.1f fps" % (n, el, n / max(el, 1e-6)))
    print("class counts:", counts)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
