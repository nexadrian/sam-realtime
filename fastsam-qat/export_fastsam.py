"""Export FastSAM-s raw LiteRT graphs using the isolated converter environment.

Example:
  python fastsam-qat/export_fastsam.py --size 512 --mode float
  python fastsam-qat/export_fastsam.py --size 512 --mode int8

The INT8 export uses camera frames listed under models/fastsam_camera/calibration.
Output artifacts and camera frames are under models/; the 512 baseline files
and comparison frames are tracked as exceptions to its default ignore rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from ultralytics import FastSAM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/FastSAM-s.pt"))
    parser.add_argument("--size", type=int, choices=(384, 512), required=True)
    parser.add_argument("--mode", choices=("float", "int8"), required=True)
    parser.add_argument("--dataset", type=Path, default=Path("models/fastsam_camera"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/fastsam_litert"))
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint missing: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"format": "litert", "imgsz": args.size, "batch": 1, "device": "cpu", "nms": False}
    if args.mode == "int8":
        images = list((args.dataset / "calibration").glob("*.jpg"))
        if not images:
            parser.error(f"No calibration JPGs in {args.dataset / 'calibration'}")
        dataset_yaml = args.dataset / "calibration.yaml"
        dataset_yaml.write_text(
            f"path: {args.dataset.resolve()}\ntrain: calibration\nval: calibration\nnames:\n  0: object\n"
        )
        kwargs.update(quantize=8, data=str(dataset_yaml.resolve()), split="val")
        print(f"Calibrating {args.size} with {len(images)} camera frames", flush=True)

    result = FastSAM(str(args.checkpoint)).export(**kwargs)
    if not result or not Path(result).is_file():
        raise RuntimeError(f"LiteRT export failed: {result}")
    destination = args.output_dir / f"FastSAM-s_{args.size}_{args.mode}.tflite"
    shutil.copy2(result, destination)
    print(destination.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
