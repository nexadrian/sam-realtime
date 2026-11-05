"""Export QAT-adapted FastSAM weights through the audited full-integer path.

The PT2E QAT graph itself contains float islands. This exporter transfers its
trained weights into the original fused float architecture, converts that
architecture to LiteRT, and calibrates a new static INT8 graph. It does not
claim to preserve PT2E fake-quant scales. Evaluate the final graph separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import litert_torch
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import qtyping, quantizer, recipe
from ultralytics import FastSAM
from ultralytics.nn.modules.head import Detect

from audit_graph import audit
from fastsam_runtime import model_input
from qat_probe import RawFastSAM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/FastSAM-s.pt"))
    parser.add_argument("--qat-weights", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(384, 512), default=512)
    parser.add_argument("--calibration", type=Path, default=Path("models/fastsam_camera/calibration"))
    parser.add_argument("--calibration-limit", type=int, help="Smoke-test with only the first N calibration images")
    parser.add_argument("--output-dir", type=Path, default=Path("models/fastsam_litert"))
    parser.add_argument("--name", required=True, help="Unique artifact label such as qat_epoch1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.calibration.glob("*.jpg"))
    if args.calibration_limit:
        paths = paths[: args.calibration_limit]
    if not paths:
        parser.error(f"No calibration images in {args.calibration}")
    model = FastSAM(str(args.checkpoint)).model.float().eval()
    model.fuse()
    saved = torch.load(args.qat_weights, map_location="cpu", weights_only=False)
    state = saved["state_dict"] if "state_dict" in saved else saved
    model_wrapper = RawFastSAM(model, args.size).eval()
    model_wrapper.load_state_dict(state, strict=True)
    for layer in model.modules():
        if isinstance(layer, Detect):
            layer.export = True
            layer.format = "litert"
    example = torch.zeros(1, 3, args.size, args.size)
    with torch.no_grad():
        model_wrapper(example)
    stem = f"FastSAM-s_{args.size}_{args.name}"
    float_path = args.output_dir / f"{stem}_float.tflite"
    int_path = args.output_dir / f"{stem}_int8.tflite"
    litert_torch.convert(model_wrapper, (example,)).export(float_path)
    print(f"Exported float: {float_path}", flush=True)

    runtime = Interpreter(model_path=str(float_path))
    runtime.allocate_tensors()
    input_detail = runtime.get_input_details()[0]
    qt = quantizer.Quantizer(str(float_path))
    qt.load_quantization_recipe(recipe.static_wi8_ai8())
    for op in (qtyping.TFLOperationName.INPUT, qtyping.TFLOperationName.OUTPUT):
        qt.update_quantization_recipe(regex=".*", operation_name=op, algorithm_key=recipe.AlgorithmName.NO_QUANTIZE)

    class Samples:
        def __iter__(self):
            for number, path in enumerate(paths, 1):
                bgr = cv2.imread(str(path))
                if bgr is None:
                    raise RuntimeError(f"Could not read {path}")
                _, tensor = model_input(bgr, input_detail)
                if number % 50 == 0:
                    print(f"Calibration sample {number}/{len(paths)}", flush=True)
                yield {"args_0": np.ascontiguousarray(tensor)}

    result = qt.calibrate({"serving_default": Samples()}, num_threads=4)
    qt.quantize(calibration_result=result, enable_progress_report=False).export_model(str(int_path), overwrite=True)
    report = audit(int_path)
    print(f"Exported INT8: {int_path}; integer audit={report['passes']}; bytes={int_path.stat().st_size}", flush=True)
    if not report["passes"]:
        raise RuntimeError(f"INT8 graph contains float or custom computation: {report['subgraphs'][0]['internal_float_operations'][:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
