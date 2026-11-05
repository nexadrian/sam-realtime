"""Prove that FastSAM-s PT2E QAT can reach a fully integer LiteRT graph.

This is an export feasibility check, not a trained accuracy result. It runs
one synthetic optimizer step with fake quantization, then exports and audits
the converted graph. Run at 512 before investing in a training dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchao.quantization.pt2e import quantize_pt2e, move_exported_model_to_eval
import litert_torch
from litert_torch.quantize import pt2e_quantizer
from ultralytics import FastSAM
from ultralytics.nn.modules.head import Detect


class RawFastSAM(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, size: int):
        super().__init__()
        self.model = model
        self.size = size

    def forward(self, image: torch.Tensor):
        proposals, prototypes = self.model(image)
        return proposals[:, :4, :] / self.size, proposals[:, 4:5, :], proposals[:, 5:, :], prototypes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/FastSAM-s.pt"))
    parser.add_argument("--size", type=int, choices=(384, 512), default=512)
    parser.add_argument("--output", type=Path, default=Path("models/fastsam_litert/FastSAM-s_512_qat_probe.tflite"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-export", action="store_true", help="Only check QAT training and reusable weights")
    parser.add_argument("--ptq-fallback", action="store_true", help="Export trained float weights and quantize remaining graph")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = FastSAM(str(args.checkpoint)).model.float().eval()
    model.fuse()
    model.requires_grad_(True)
    batch_norms = sum(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.modules())
    print(f"remaining BatchNorm2d modules={batch_norms}", flush=True)
    for layer in model.modules():
        if isinstance(layer, Detect):
            layer.export = True
            layer.format = "litert"
    wrapper = RawFastSAM(model, args.size).eval()
    example = torch.rand(1, 3, args.size, args.size)
    # Populate the fixed-size anchors before export. Tracing anchor creation
    # makes stride scalars data-dependent in the second LiteRT export pass.
    with torch.no_grad():
        wrapper(example)
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
    prepared = quantize_pt2e.prepare_qat_pt2e(exported, quantizer).to(device)
    print("prepare_qat_pt2e succeeded", flush=True)
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
        from ai_edge_quantizer import qtyping, quantizer as edge_quantizer, recipe
        from ai_edge_litert.interpreter import Interpreter
        import cv2
        from fastsam_runtime import model_input

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
    prepared = prepared.cpu()
    move_exported_model_to_eval(prepared)
    converted = quantize_pt2e.convert_pt2e(prepared, fold_quantize=False)
    print("convert_pt2e succeeded", flush=True)
    nodes = list(converted.graph.nodes)
    for index, node in enumerate(nodes):
        if "_local_scalar_dense" in str(node.target):
            print("SCALAR NODE", *[str(item) for item in nodes[max(0, index - 4): index + 2]], sep="\n", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    litert_torch.convert(converted, (example,)).export(args.output)
    print(f"LiteRT export succeeded: {args.output}", flush=True)
    from audit_graph import audit
    report = audit(args.output)
    print(f"Integer audit: passes={report['passes']}", flush=True)
    for graph in report["subgraphs"]:
        print(f"operators={graph['operators']} float_internal={len(graph['internal_float_operations'])}", flush=True)
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
