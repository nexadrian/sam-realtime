"""Compare FastSAM LiteRT masks with PyTorch on held-out camera frames.

This validation tool uses Ultralytics only in the converter environment.
The runtime module itself has no Ultralytics or PyTorch dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
from ultralytics import FastSAM

from fastsam_runtime import FastSamTFLite, crop_and_resize


def torch_masks(model: FastSAM, frame: np.ndarray) -> np.ndarray:
    size = frame.shape[0]
    result = model(
        frame,
        device="cpu",
        retina_masks=True,
        imgsz=size,
        conf=0.4,
        iou=0.9,
        points=[[size // 2, size // 2]],
        labels=[1],
        verbose=False,
    )[0]
    if result.masks is None:
        return np.empty((0, size, size), dtype=np.uint8)
    return result.masks.data.cpu().numpy().astype(np.uint8)


def same_size(masks: np.ndarray, size: int) -> np.ndarray:
    if masks.shape[1] == size:
        return masks
    return np.stack([cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) for mask in masks]) if len(masks) else np.empty((0, size, size), dtype=np.uint8)


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    candidate = same_size(candidate, reference.shape[1])
    count_match = len(reference) == len(candidate)
    if not len(reference) and not len(candidate):
        return {"reference_count": 0, "candidate_count": 0, "count_match": True, "ious": [], "paired_ious": [], "union_iou": None}
    reference_union = np.any(reference, axis=0) if len(reference) else np.zeros(reference.shape[1:], dtype=bool)
    candidate_union = np.any(candidate, axis=0) if len(candidate) else np.zeros(reference.shape[1:], dtype=bool)
    union_pixels = np.count_nonzero(reference_union | candidate_union)
    union_iou = np.count_nonzero(reference_union & candidate_union) / union_pixels if union_pixels else 1.0
    pairs = []
    for i, left in enumerate(reference):
        for j, right in enumerate(candidate):
            intersection = np.count_nonzero(left & right)
            union = np.count_nonzero(left | right)
            pairs.append((intersection / union if union else 1.0, i, j))
    used_left, used_right, ious = set(), set(), []
    for iou, i, j in sorted(pairs, reverse=True):
        if i not in used_left and j not in used_right:
            ious.append(float(iou))
            used_left.add(i)
            used_right.add(j)
    # Missing masks contribute zero rather than disappearing from the median.
    paired_ious = ious.copy()
    ious.extend([0.0] * (max(len(reference), len(candidate)) - len(ious)))
    return {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "count_match": count_match,
        "ious": ious,
        "paired_ious": paired_ious,
        "union_iou": float(union_iou),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/FastSAM-s.pt"))
    parser.add_argument("--frames", type=Path, default=Path("models/fastsam_camera/holdout"))
    parser.add_argument("--models", type=Path, default=Path("models/fastsam_litert"))
    parser.add_argument("--output", type=Path, default=Path("models/fastsam_litert/evaluation.json"))
    parser.add_argument("--limit", type=int, help="Use only the first N hold-out images for a smoke run")
    parser.add_argument("--float-only", action="store_true", help="Validate float exports before quantization")
    parser.add_argument("--variant-512", choices=("stock", "split"), default="stock")
    parser.add_argument("--variant-384", choices=("stock", "split"), default="stock")
    args = parser.parse_args()
    frames = sorted(args.frames.glob("*.jpg"))
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        parser.error(f"No hold-out JPGs found in {args.frames}")

    modes = ("float",) if args.float_only else ("float", "int8")
    variant = {512: args.variant_512, 384: args.variant_384}
    model_paths = {
        f"{size}_{mode}": args.models / f"FastSAM-s_{size}{'_split' if variant[size] == 'split' else ''}_{mode}.tflite"
        for size in (512, 384) for mode in modes
    }
    missing = [path for path in model_paths.values() if not path.is_file()]
    if missing:
        parser.error(f"Missing models: {missing}")
    runtimes = {key: FastSamTFLite(path) for key, path in model_paths.items()}
    pytorch = FastSAM(str(args.checkpoint))
    comparisons = {key: [] for key in ("512_float_vs_torch", "384_float_vs_torch512")}
    if not args.float_only:
        comparisons.update({key: [] for key in ("512_int8_vs_float", "384_int8_vs_float", "384_int8_vs_torch512")})
    timings = {key: [] for key in runtimes}
    failures = []

    for number, path in enumerate(frames, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read {path}")
        prepared = crop_and_resize(frame, 512)
        baseline = torch_masks(pytorch, prepared)
        predictions = {}
        for key, runtime in runtimes.items():
            start = time.perf_counter()
            predictions[key] = runtime.infer(frame)
            timings[key].append((time.perf_counter() - start) * 1000)
        matches = {
            "512_float_vs_torch": compare(baseline, predictions["512_float"]),
            "384_float_vs_torch512": compare(baseline, predictions["384_float"]),
        }
        if not args.float_only:
            matches.update({
                "512_int8_vs_float": compare(predictions["512_float"], predictions["512_int8"]),
                "384_int8_vs_float": compare(predictions["384_float"], predictions["384_int8"]),
                "384_int8_vs_torch512": compare(baseline, predictions["384_int8"]),
            })
        for key, item in matches.items():
            comparisons[key].append(item)
        if any(not item["count_match"] or any(iou < 0.85 for iou in item["ious"]) for item in matches.values()):
            failures.append({"frame": path.name, "comparisons": matches})
        if number % 10 == 0 or number == len(frames):
            print(f"Evaluated {number}/{len(frames)}", flush=True)

    summary = {}
    for key, items in comparisons.items():
        ious = [value for item in items for value in item["ious"]]
        paired_ious = [value for item in items for value in item["paired_ious"]]
        union_ious = [item["union_iou"] for item in items if item["union_iou"] is not None]
        summary[key] = {
            "count_agreement": sum(item["count_match"] for item in items) / len(items),
            "median_mask_iou": statistics.median(ious) if ious else None,
            "mask_pairs": len(ious),
            "median_paired_mask_iou": statistics.median(paired_ious) if paired_ious else None,
            "paired_mask_pairs": len(paired_ious),
            "median_union_mask_iou_nonempty": statistics.median(union_ious) if union_ious else None,
            "empty_both_frames": sum(item["reference_count"] == item["candidate_count"] == 0 for item in items),
        }
    report = {
        "frames": len(frames),
        "variant": {str(size): name for size, name in variant.items()},
        "comparisons": summary,
        "acceptance": {
            key: (summary[key]["count_agreement"] >= 0.95 and (summary[key]["median_mask_iou"] or 0.0) >= 0.85)
            for key in ("512_int8_vs_float", "384_int8_vs_float") if key in summary
        },
        "inference_ms_median": {key: statistics.median(values) for key, values in timings.items()},
        "model_bytes": {key: path.stat().st_size for key, path in model_paths.items()},
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
