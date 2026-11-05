"""Probe FastSAM-s PT2E QAT training and integer LiteRT export.

This is an export feasibility check, not a trained accuracy result. It runs
one synthetic optimizer step with fake quantization, then exports and audits
the converted graph. Some variants fail the integer audit; the raw-head,
explicit-SiLU variant is the one that reaches an integer graph. Run at 512
before investing in a training dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType

import torch
from torchao.quantization.pt2e import quantize_pt2e, move_exported_model_to_eval
import litert_torch
from litert_torch.quantize import pt2e_quantizer
from litert_torch.quantize.quant_config import QuantConfig
from ultralytics import FastSAM
from ultralytics.nn.modules.head import Detect, Segment


class RawFastSAM(torch.nn.Module):
    """Return decoded boxes, scores, mask coefficients, and prototypes.

    Normalize boxes to the input size so the exported outputs have the same
    shape and meaning as the application runtime expects.
    """

    def __init__(self, model: torch.nn.Module, size: int):
        super().__init__()
        self.model = model
        self.size = size

    def forward(self, image: torch.Tensor):
        proposals, prototypes = self.model(image)
        return proposals[:, :4, :] / self.size, proposals[:, 4:5, :], proposals[:, 5:, :], prototypes


class RawHeadFastSAM(torch.nn.Module):
    """Return head logits and prototypes; decode boxes and scores at runtime."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        return self.model(image)


def raw_segment_forward(head: Segment, features: list[torch.Tensor]):
    """Bypass DFL box decoding and score sigmoid in the exported graph."""
    predictions = head.forward_head(features, **head.one2many)
    return (
        predictions["boxes"],
        predictions["scores"],
        predictions["mask_coefficient"],
        head.proto(features[0]),
    )


def enable_raw_head(model: torch.nn.Module) -> None:
    """Replace the final segmentation head's forward method with raw output."""
    head = model.model[-1]
    if not isinstance(head, Segment):
        raise TypeError(f"Expected a Segment head, got {type(head).__name__}")
    head.forward = MethodType(raw_segment_forward, head)


class ExplicitSiLU(torch.nn.Module):
    """Expose SiLU as sigmoid and multiplication to the PT2E quantizer."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(value)


def replace_silu(module: torch.nn.Module) -> int:
    """Expose every SiLU as sigmoid and multiply without changing weights."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.SiLU):
            setattr(module, name, ExplicitSiLU())
            replaced += 1
        else:
            replaced += replace_silu(child)
    return replaced


def main() -> int:
    """Prepare QAT, take one training step, export, and audit the graph."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/FastSAM-s.pt"))
    parser.add_argument("--size", type=int, choices=(384, 512), default=512)
    parser.add_argument("--output", type=Path, default=Path("models/fastsam_litert/FastSAM-s_512_qat_probe.tflite"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-export", action="store_true", help="Only check QAT training and reusable weights")
    parser.add_argument("--ptq-fallback", action="store_true", help="Export trained float weights and quantize remaining graph")
    parser.add_argument("--fold-quantize", action="store_true", help="Fold quantized constants before LiteRT export")
    parser.add_argument("--explicit-silu", action="store_true", help="Rewrite parameter-free SiLU as sigmoid and multiply before QAT")
    parser.add_argument("--raw-head", action="store_true", help="Export raw head tensors and decode boxes/scores outside LiteRT")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = FastSAM(str(args.checkpoint)).model.float().eval()
    # Fold Conv+BatchNorm before tracing. The fused structure is the one QAT
    # sees, and it avoids converter patterns that are inapplicable here.
    model.fuse()
    if args.explicit_silu:
        print(f"replaced SiLU modules={replace_silu(model)}", flush=True)
    if args.raw_head:
        enable_raw_head(model)
    model.requires_grad_(True)
    batch_norms = sum(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.modules())
    print(f"remaining BatchNorm2d modules={batch_norms}", flush=True)
    for layer in model.modules():
        if isinstance(layer, Detect):
            layer.export = True
            layer.format = "litert"
    wrapper = (RawHeadFastSAM(model) if args.raw_head else RawFastSAM(model, args.size)).eval()
    example = torch.rand(1, 3, args.size, args.size)
    # Populate the fixed-size anchors before export. Tracing anchor creation
    # makes stride scalars data-dependent in the second LiteRT export pass.
    with torch.no_grad():
        wrapper(example)
    # PT2E QAT operates on an exported, fixed-shape PyTorch graph rather than
    # directly on the Ultralytics module tree.
    exported = torch.export.export(wrapper, (example,)).module()
    print("torch.export succeeded", flush=True)
    if batch_norms == 0:
        # LiteRT Torch 0.9.4's QAT Conv+BN pattern builder tries to export a
        # Python function as a Module with torchao 0.18. The fused FastSAM
        # graph has no BN, so those patterns are inapplicable here.
        pt2e_quantizer.PT2EQuantizer.STATIC_QAT_ONLY_OPS = []
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_qat=True, is_per_channel=True)
    )
    # Insert observers and fake-quant operations. Training still uses float
    # tensors; fake quantization simulates the rounding/clipping of INT8.
    prepared = quantize_pt2e.prepare_qat_pt2e(exported, quantizer).to(device)
    print("prepare_qat_pt2e succeeded", flush=True)
    # This arbitrary loss only checks forward/backward and weight updates.
    # It is not the teacher-guided training objective used by qat_train.py.
    optimizer = torch.optim.SGD(prepared.parameters(), lr=1e-5)
    optimizer.zero_grad()
    outputs = prepared(example.to(device))
    loss = outputs[1].mean() + outputs[3].square().mean()
    loss.backward()
    optimizer.step()
    print(f"one QAT optimizer step succeeded: loss={float(loss.detach()):.6f}", flush=True)
    original_names = set(wrapper.state_dict())
    trained_names = set(prepared.state_dict())
    print(f"reusable weight keys={len(original_names & trained_names)}/{len(original_names)}", flush=True)
    print(f"missing examples={list(sorted(original_names - trained_names))[:5]}", flush=True)
    if args.skip_export:
        return 0
    if args.ptq_fallback:
        # Transfer updated float weights out of the fake-quant graph, export
        # a float model, then calibrate a separate INT8 graph from images.
        # This does not preserve the QAT observer scales or zero points.
        from ai_edge_quantizer import qtyping, quantizer as edge_quantizer, recipe
        from ai_edge_litert.interpreter import Interpreter
        import cv2
        from sam_realtime.segmenters.fastsam_tflite import model_input

        trained = prepared.state_dict()
        wrapper.load_state_dict({key: trained[key].detach().cpu() for key in original_names})
        wrapper = wrapper.cpu().eval()
        float_path = args.output.with_name(args.output.stem + "_float.tflite")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        litert_torch.convert(wrapper, (example,)).export(float_path)
        qt = edge_quantizer.Quantizer(str(float_path))
        qt.load_quantization_recipe(recipe.static_wi8_ai8())
        for op in (qtyping.TFLOperationName.INPUT, qtyping.TFLOperationName.OUTPUT):
            qt.update_quantization_recipe(regex=".*", operation_name=op, algorithm_key=recipe.AlgorithmName.NO_QUANTIZE)
        runtime = Interpreter(model_path=str(float_path))
        runtime.allocate_tensors()
        # Three frames keep the fallback probe short; real PTQ uses the full
        # calibration set and must be evaluated on separate holdout frames.
        paths = sorted(Path("models/fastsam_camera/calibration").glob("*.jpg"))[:3]
        samples = []
        for path in paths:
            frame = cv2.imread(str(path))
            _, tensor = model_input(frame, runtime.get_input_details()[0])
            samples.append({"args_0": tensor})
        qsv = qt.calibrate({"serving_default": samples}, num_threads=4)
        qt.quantize(calibration_result=qsv, enable_progress_report=False).export_model(str(args.output), overwrite=True)
        from audit_graph import audit
        report = audit(args.output)
        print(f"QAT-weight PTQ export: {args.output}; integer audit={report['passes']}", flush=True)
        return 0 if report["passes"] else 1
    # Switch the graph to inference mode, replace fake quantization with
    # explicit Q/DQ edges, then use the LiteRT Torch PT2E lowering path.
    prepared = prepared.cpu()
    move_exported_model_to_eval(prepared)
    converted = quantize_pt2e.convert_pt2e(prepared, fold_quantize=args.fold_quantize)
    print("convert_pt2e succeeded", flush=True)
    nodes = list(converted.graph.nodes)
    for index, node in enumerate(nodes):
        if "_local_scalar_dense" in str(node.target):
            print("SCALAR NODE", *[str(item) for item in nodes[max(0, index - 4): index + 2]], sep="\n", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # The converter needs the PT2E config to lower Q/DQ patterns to integer
    # LiteRT operators. Without it, the graph retains float computation.
    litert_torch.convert(
        converted, (example,), quant_config=QuantConfig(pt2e_quantizer=quantizer)
    ).export(args.output)
    print(f"LiteRT export succeeded: {args.output}", flush=True)
    from audit_graph import audit
    from ai_edge_litert.interpreter import Interpreter

    report = audit(args.output)
    # Integer structure and runtime validity are separate checks. In
    # particular, an integer SOFTMAX may still have invalid output qparams.
    try:
        interpreter = Interpreter(model_path=str(args.output))
        interpreter.allocate_tensors()
        report["interpreter_allocation"] = {"passes": True}
    except Exception as error:
        report["interpreter_allocation"] = {"passes": False, "error": str(error)}
    report_path = args.output.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Integer audit: passes={report['passes']}", flush=True)
    for graph in report["subgraphs"]:
        print(f"operators={graph['operators']} float_internal={len(graph['internal_float_operations'])}", flush=True)
    print(f"Interpreter allocation: {report['interpreter_allocation']}; report={report_path}", flush=True)
    return 0 if report["passes"] and report["interpreter_allocation"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
