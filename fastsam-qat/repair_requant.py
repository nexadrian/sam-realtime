"""Restore PT2E requantization lost at chunk-to-convolution edges in LiteRT.

The direct QAT converter can connect an INT8 slice directly to a convolution
even when the PT2E graph requantizes that slice first. The convolution bias
then uses the PT2E input scale, while the actual LiteRT input has another
scale. This script restores the missing INT8-to-INT8 QUANTIZE operations in a
copy of a direct-QAT model. It requires the original full QAT state so both
the scale and zero point come from the trained graph.

This repair assumes the PT2E and LiteRT graphs have the same convolution order.
It checks that assumption against each convolution's bias scale before editing
the FlatBuffer, then verifies the repaired graph and writes a diagnostic report.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter
import flatbuffers
import numpy as np
import torch
from torchao.quantization.pt2e import move_exported_model_to_eval, quantize_pt2e
from litert_torch.quantize import pt2e_quantizer
from ultralytics import FastSAM
from ultralytics.nn.modules.head import Detect

from audit_graph import audit
from qat_probe import RawHeadFastSAM, enable_raw_head, replace_silu


def pt2e_convolution_inputs(checkpoint: Path, state: Path, size: int) -> list[tuple[float, int] | None]:
    """Recover each convolution's intended input scale and zero point from QAT.

    Entries follow graph execution order. Transposed convolutions occupy a
    position in that order but return ``None`` because this repair only changes
    ordinary ``CONV_2D`` inputs.
    """
    saved = torch.load(state, map_location="cpu", weights_only=False)
    if not saved.get("explicit_silu") or not saved.get("raw_head") or saved.get("size") != size:
        raise ValueError("QAT state must use explicit SiLU, raw head, and the requested input size")
    model = FastSAM(str(checkpoint)).model.float().eval()
    model.fuse()
    if replace_silu(model) != 66:
        raise RuntimeError("Expected 66 SiLU modules")
    enable_raw_head(model)
    model.requires_grad_(True)
    for layer in model.modules():
        if isinstance(layer, Detect):
            layer.export = True
            layer.format = "litert"
    wrapper = RawHeadFastSAM(model).eval()
    example = torch.zeros(1, 3, size, size)
    # Populate fixed-size anchors before export, as in the original QAT run.
    with torch.no_grad():
        wrapper(example)
    exported = torch.export.export(wrapper, (example,)).module()
    pt2e_quantizer.PT2EQuantizer.STATIC_QAT_ONLY_OPS = []
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_qat=True, is_per_channel=True)
    )
    prepared = quantize_pt2e.prepare_qat_pt2e(exported, quantizer)
    prepared.load_state_dict(saved["state_dict"], strict=True)
    move_exported_model_to_eval(prepared)
    converted = quantize_pt2e.convert_pt2e(prepared, fold_quantize=False)
    # A dequantize node immediately before conv records the activation's
    # trained INT8 representation. Its arguments contain scale and zero point.
    inputs = []
    for node in converted.graph.nodes:
        if node.op != "call_function" or not (
            "conv2d.default" in str(node.target) or "conv_transpose2d" in str(node.target)
        ):
            continue
        if "conv_transpose2d" in str(node.target):
            inputs.append(None)
            continue
        source = node.args[0]
        if "dequantize_per_tensor.default" not in str(source.target):
            raise RuntimeError(f"Unexpected convolution input: {source.target}")
        inputs.append((float(source.args[1]), int(source.args[2])))
    return inputs


def convolution_scale_errors(graph, codes) -> list[dict]:
    """Report convolutions whose bias scale disagrees with their actual input.

    For each output channel, the INT32 bias scale must equal the INT8 input
    scale times that channel's weight scale. The ratio bias/weight reveals the
    input scale for which the bias was quantized.
    """
    errors = []
    for op_index, op in enumerate(graph.operators):
        if codes[op.opcodeIndex].builtinCode != schema.BuiltinOperator.CONV_2D:
            continue
        image, weight, bias = (graph.tensors[int(i)] for i in op.inputs[:3])
        image_scale = float(image.quantization.scale[0])
        weight_scales = np.asarray(weight.quantization.scale, dtype=np.float64)
        bias_scales = np.asarray(bias.quantization.scale, dtype=np.float64)
        expected = image_scale * weight_scales
        relative = np.max(np.abs(bias_scales / expected - 1))
        if relative > 1e-4:
            errors.append({"operator": op_index, "input_scale": image_scale,
                           "bias_expects_input_scale": float(np.median(bias_scales / weight_scales)),
                           "max_relative_error": float(relative)})
    return errors


def main() -> int:
    """Insert missing INT8 requantization edges and validate the new model."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qat-state", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/FastSAM-s.pt"))
    parser.add_argument("--size", type=int, choices=(384, 512), default=512)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must be a new file; keep the original export")

    # Rebuild PT2E from the full QAT state: a weights-only checkpoint cannot
    # recover the observers' activation scales and zero points.
    targets = pt2e_convolution_inputs(args.checkpoint, args.qat_state, args.size)
    raw = args.input.read_bytes()
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAs(raw, 0))
    if len(model.subgraphs) != 1:
        raise ValueError("Expected one LiteRT subgraph")
    graph = model.subgraphs[0]
    quant_opcode = next(
        (i for i, code in enumerate(model.operatorCodes)
         if code.builtinCode == schema.BuiltinOperator.QUANTIZE), None
    )
    if quant_opcode is None:
        raise ValueError("Model has no INT8 QUANTIZE operator to copy")
    # Reuse an existing operator record so the inserted edge has the model's
    # QUANTIZE opcode and options; its input and output tensors are set below.
    template = next(op for op in graph.operators if op.opcodeIndex == quant_opcode)
    before = convolution_scale_errors(graph, model.operatorCodes)
    repaired = []
    operations = []
    conv_number = 0
    # Match both ordinary and transposed convolutions to PT2E in execution
    # order. Bias scales must corroborate the match before any edge is changed.
    for op_index, op in enumerate(graph.operators):
        kind = model.operatorCodes[op.opcodeIndex].builtinCode
        if kind in (schema.BuiltinOperator.CONV_2D, schema.BuiltinOperator.TRANSPOSE_CONV):
            if conv_number >= len(targets):
                raise RuntimeError("LiteRT has more convolutions than PT2E")
            target = targets[conv_number]
            conv_number += 1
            if kind == schema.BuiltinOperator.CONV_2D:
                image, weight, bias = (graph.tensors[int(i)] for i in op.inputs[:3])
                current_scale = float(image.quantization.scale[0])
                weight_scales = np.asarray(weight.quantization.scale, dtype=np.float64)
                bias_scales = np.asarray(bias.quantization.scale, dtype=np.float64)
                bias_input_scale = float(np.median(bias_scales / weight_scales))
                if target is None or not np.isclose(bias_input_scale, target[0], rtol=1e-5):
                    raise RuntimeError(f"PT2E/LiteRT convolution order or scales differ at {op_index}")
                if not np.isclose(current_scale, target[0], rtol=1e-5):
                    source_index = int(op.inputs[0])
                    # Keep the original producer tensor for its other users.
                    # The new tensor holds the producer's values requantized
                    # to the scale expected by this convolution's INT32 bias.
                    tensor = copy.deepcopy(image)
                    tensor.name = (image.name or b"tensor") + f"_requant_for_op_{op_index}".encode()
                    tensor.buffer = 0
                    tensor.quantization.scale = [target[0]]
                    tensor.quantization.zeroPoint = [target[1]]
                    tensor.quantization.min = None
                    tensor.quantization.max = None
                    target_index = len(graph.tensors)
                    graph.tensors.append(tensor)
                    # LiteRT QUANTIZE supports INT8 input and INT8 output with
                    # different scales and zero points. Insert it immediately
                    # before this convolution and reconnect only that input.
                    quantize = copy.deepcopy(template)
                    quantize.inputs = [source_index]
                    quantize.outputs = [target_index]
                    operations.append(quantize)
                    op.inputs = [target_index, *[int(value) for value in op.inputs[1:]]]
                    repaired.append({"operator": op_index, "source_tensor": source_index,
                                     "new_tensor": target_index, "source_scale": current_scale,
                                     "target_scale": target[0], "target_zero_point": target[1]})
        operations.append(op)
    if conv_number != len(targets):
        raise RuntimeError("PT2E has more convolutions than LiteRT")
    if not repaired:
        raise RuntimeError("No missing requantization edges found")
    graph.operators = operations
    # The graph must satisfy the bias-scale invariant at every convolution,
    # including ones that did not need a new edge.
    after = convolution_scale_errors(graph, model.operatorCodes)
    if after:
        raise RuntimeError(f"Convolution bias scales still mismatch after repair: {after}")
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(builder.Output())
    # An all-integer audit alone cannot catch scale mismatches. Also ensure
    # LiteRT accepts the rewritten FlatBuffer and records the before/after data.
    interpreter = Interpreter(model_path=str(args.output))
    interpreter.allocate_tensors()
    graph_report = audit(args.output)
    if not graph_report["passes"]:
        raise RuntimeError("Repaired model contains float or custom operations")
    report = {"source": str(args.input), "output": str(args.output), "before": before,
              "repaired": repaired, "remaining_scale_errors": after,
              "integer_graph": graph_report["passes"]}
    args.output.with_suffix(".repair.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Repaired {len(repaired)} convolution inputs; LiteRT allocation and scale audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
