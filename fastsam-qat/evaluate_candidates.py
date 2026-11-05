"""Compare arbitrary INT8 FastSAM candidates to the float graph on saved frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import cv2

from evaluate import compare
from fastsam_runtime import FastSamTFLite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--float-model", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=MODEL")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = {}
    for item in args.candidate:
        name, separator, path = item.partition("=")
        if not separator or not name or name in models:
            parser.error(f"Expected distinct NAME=MODEL, got {item!r}")
        models[name] = Path(path)
    paths = sorted(args.frames.glob("*.jpg"))[args.start:]
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error(f"No frames selected from {args.frames}")
    reference = FastSamTFLite(args.float_model)
    candidates = {name: FastSamTFLite(path) for name, path in models.items()}
    comparisons = {name: [] for name in candidates}
    for number, path in enumerate(paths, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read {path}")
        masks = reference.infer(frame)
        for name, runtime in candidates.items():
            comparisons[name].append(compare(masks, runtime.infer(frame)))
        if number % 10 == 0:
            print(f"Compared {number}/{len(paths)}", flush=True)
    summary = {}
    for name, values in comparisons.items():
        all_ious = [iou for value in values for iou in value["ious"]]
        paired_ious = [iou for value in values for iou in value["paired_ious"]]
        unions = [value["union_iou"] for value in values if value["union_iou"] is not None]
        summary[name] = {
            "count_agreement": sum(value["count_match"] for value in values) / len(values),
            "median_mask_iou": statistics.median(all_ious) if all_ious else None,
            "median_paired_mask_iou": statistics.median(paired_ious) if paired_ious else None,
            "median_union_mask_iou": statistics.median(unions) if unions else None,
            "mask_pairs": len(all_ious),
            "per_frame": [{"name": path.name, **value} for path, value in zip(paths, values)],
        }
    report = {"frames": [path.name for path in paths], "float_model": str(args.float_model),
              "models": {name: str(path) for name, path in models.items()}, "comparisons": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in data.items() if key != "per_frame"}
                      for name, data in summary.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
