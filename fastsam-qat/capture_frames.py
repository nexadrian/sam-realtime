"""Capture local camera frames for FastSAM calibration or hold-out evaluation.

Example:
  python tflite/capture_frames.py --split calibration --count 350
  python tflite/capture_frames.py --split holdout --count 100

The split is chosen before capture so adjacent frames cannot leak between
calibration and evaluation. Image data stays under the git-ignored models/ dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2


def center_square(frame):
    height, width = frame.shape[:2]
    side = min(height, width)
    x0, y0 = (width - side) // 2, (height - side) // 2
    return frame[y0 : y0 + side, x0 : x0 + side]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--split", choices=("calibration", "holdout"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--interval", type=float, default=0.8, help="Seconds between saved frames")
    parser.add_argument("--output-dir", type=Path, default=Path("models/fastsam_camera"))
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.interval <= 0:
        parser.error("--count and --interval must both be positive")

    split_dir = args.output_dir / args.split
    if split_dir.exists() and any(split_dir.glob("*.jpg")):
        raise RuntimeError(f"{split_dir} already has images; choose a fresh output directory")
    split_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open OpenCV camera index {args.camera}")

    manifest = {"camera": args.camera, "split": args.split, "interval_s": args.interval, "frames": []}
    print(f"Camera {args.camera}: collecting {args.count} {args.split} frames.", flush=True)
    print("Vary objects, distance, lighting, overlap, and whether the center is empty.", flush=True)
    print(
        "Use Ctrl+C to stop early." if args.no_preview else "Press q in the preview to stop early, or Ctrl+C in the terminal.",
        flush=True,
    )
    next_save = time.monotonic() + 5.0
    try:
        while len(manifest["frames"]) < args.count:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")
            square = center_square(frame)
            now = time.monotonic()
            if now >= next_save:
                number = len(manifest["frames"]) + 1
                path = split_dir / f"{number:04d}.jpg"
                if not cv2.imwrite(str(path), square, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    raise RuntimeError(f"Could not write {path}")
                manifest["frames"].append({
                    "file": path.name,
                    "timestamp": time.time(),
                    "size": list(square.shape[:2]),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
                if number % 25 == 0 or number == args.count:
                    print(f"{args.split}: {number}/{args.count}", flush=True)
                next_save = now + args.interval
            if not args.no_preview:
                preview = cv2.resize(square, (512, 512), interpolation=cv2.INTER_AREA)
                cv2.drawMarker(preview, (256, 256), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
                label = f"{args.split} {len(manifest['frames'])}/{args.count}"
                cv2.putText(preview, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("FastSAM dataset capture", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("Capture interrupted; saving the partial manifest.", flush=True)
    finally:
        capture.release()
        cv2.destroyAllWindows()
        manifest["count"] = len(manifest["frames"])
        (args.output_dir / f"{args.split}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {manifest['count']} frames to {split_dir}", flush=True)
    return 0 if manifest["count"] == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
