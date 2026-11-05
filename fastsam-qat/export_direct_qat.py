"""Export a trained raw-head PT2E QAT checkpoint directly to integer LiteRT.

The model outputs raw DFL box distributions, score logits, mask coefficients,
and prototypes. The application LiteRT runtime decodes them outside the integer graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_edge_litert.interpreter import Interpreter
import litert_torch
from litert_torch.quantize import pt2e_quantizer
from litert_torch.quantize.quant_config import QuantConfig
import torch
from torchao.quantization.pt2e import move_exported_model_to_eval, quantize_pt2e
from ultralytics import FastSAM
from ultralytics.nn.modules.head import Detect

from audit_graph import audit
from qat_probe import RawHeadFastSAM, enable_raw_head, replace_silu


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/FastSAM-s.pt"))
    parser.add_argument("--qat-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(384, 512), default=512)
    parser.add_argument("--fold-quantize", action="store_true", help="Fold quantized constants before conversion")
    args = parser.parse_args()

    saved = torch.load(args.qat_state, map_location="cpu", weights_only=False)
    if not saved.get("explicit_silu") or not saved.get("raw_head") or saved.get("size") != args.size:
        parser.error("QAT state must use explicit SiLU, raw head, and the requested input size")

    model = FastSAM(str(args.checkpoint)).model.float().eval()
    model.fuse()
    if replace_silu(model) != 66:
        raise RuntimeError("Expected 66 parameter-free SiLU modules")
    enable_raw_head(model)
    model.requires_grad_(True)
    for layer in model.modules():
        if isinstance(layer, Detect):
            layer.export = True
            layer.format = "litert"
    wrapper = RawHeadFastSAM(model).eval()
    example = torch.zeros(1, 3, args.size, args.size)
    with torch.no_grad():
        wrapper(example)
    exported = torch.export.export(wrapper, (example,)).module()
    # BatchNorm has been fused, so these Conv+BN-only QAT patterns are unused.
    assert not any(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.modules())
    pt2e_quantizer.PT2EQuantizer.STATIC_QAT_ONLY_OPS = []
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_qat=True, is_per_channel=True)
    )
    prepared = quantize_pt2e.prepare_qat_pt2e(exported, quantizer)
    prepared.load_state_dict(saved["state_dict"], strict=True)
    move_exported_model_to_eval(prepared)
    converted = quantize_pt2e.convert_pt2e(prepared, fold_quantize=args.fold_quantize)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    litert_torch.convert(
        converted, (example,), quant_config=QuantConfig(pt2e_quantizer=quantizer)
    ).export(args.output)

    report = audit(args.output)
    report_path = args.output.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passes"]:
        raise RuntimeError(f"Internal float operations remain: {report_path}")
    interpreter = Interpreter(model_path=str(args.output))
    interpreter.allocate_tensors()
    input_shape = interpreter.get_input_details()[0]["shape"].tolist()
    output_shapes = [detail["shape"].tolist() for detail in interpreter.get_output_details()]
    print(f"Direct QAT export: {args.output}; bytes={args.output.stat().st_size}; "
          f"input={input_shape}; outputs={output_shapes}; integer audit passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
