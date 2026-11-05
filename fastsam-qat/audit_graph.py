"""Audit a LiteRT FlatBuffer for built-in ops and integer internal compute.

Run with the converter environment, which provides the FlatBuffer schema.
The runtime does not import this module or need its dependencies.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from ai_edge_litert import schema_py_generated as schema


def _name(enum, value: int) -> str:
    return next((key for key, number in vars(enum).items() if number == value), str(value))


def _tensor_info(index: int, tensor) -> dict:
    quant = tensor.Quantization()
    return {
        "index": index,
        "name": tensor.Name().decode() if tensor.Name() else None,
        "shape": [tensor.Shape(i) for i in range(tensor.ShapeLength())],
        "type": _name(schema.TensorType, tensor.Type()),
        "scales": [quant.Scale(i) for i in range(quant.ScaleLength())] if quant else [],
        "zero_points": [quant.ZeroPoint(i) for i in range(quant.ZeroPointLength())] if quant else [],
    }


def audit(path: Path) -> dict:
    data = path.read_bytes()
    if not schema.Model.ModelBufferHasIdentifier(data, 0):
        raise ValueError(f"Not a TFLite FlatBuffer: {path}")
    model = schema.Model.GetRootAs(data, 0)
    codes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]
    report = {"model": str(path), "bytes": len(data), "subgraphs": [], "passes": True}
    float_types = {schema.TensorType.FLOAT16, schema.TensorType.FLOAT32, schema.TensorType.FLOAT64, schema.TensorType.BFLOAT16}

    for graph_index in range(model.SubgraphsLength()):
        graph = model.Subgraphs(graph_index)
        tensors = [graph.Tensors(i) for i in range(graph.TensorsLength())]
        graph_inputs = {graph.Inputs(i) for i in range(graph.InputsLength())}
        graph_outputs = {graph.Outputs(i) for i in range(graph.OutputsLength())}
        operations = Counter()
        custom = []
        float_ops = []
        for index in range(graph.OperatorsLength()):
            op = graph.Operators(index)
            code = codes[op.OpcodeIndex()]
            name = _name(schema.BuiltinOperator, code.BuiltinCode())
            operations[name] += 1
            if code.BuiltinCode() == schema.BuiltinOperator.CUSTOM:
                custom.append({"index": index, "name": code.CustomCode().decode() if code.CustomCode() else "CUSTOM"})

            inputs = [op.Inputs(i) for i in range(op.InputsLength()) if op.Inputs(i) >= 0]
            outputs = [op.Outputs(i) for i in range(op.OutputsLength()) if op.Outputs(i) >= 0]
            float_inputs = [i for i in inputs if tensors[i].Type() in float_types]
            float_outputs = [i for i in outputs if tensors[i].Type() in float_types]
            if float_inputs or float_outputs:
                allowed = (
                    name == "QUANTIZE" and all(i in graph_inputs for i in float_inputs) and not float_outputs
                ) or (
                    name == "DEQUANTIZE" and not float_inputs and all(i in graph_outputs for i in float_outputs)
                )
                if not allowed:
                    float_ops.append({
                        "index": index,
                        "op": name,
                        "float_inputs": float_inputs,
                        "float_outputs": float_outputs,
                    })
        tensor_types = Counter(_name(schema.TensorType, tensor.Type()) for tensor in tensors)
        item = {
            "index": graph_index,
            "inputs": [_tensor_info(i, tensors[i]) for i in sorted(graph_inputs)],
            "outputs": [_tensor_info(i, tensors[i]) for i in sorted(graph_outputs)],
            "tensor_types": dict(tensor_types),
            "operators": dict(operations),
            "custom_operators": custom,
            "internal_float_operations": float_ops,
            "passes": not custom and not float_ops,
        }
        report["subgraphs"].append(item)
        report["passes"] &= item["passes"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.model)
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")
    print(json.dumps({
        "model": report["model"],
        "bytes": report["bytes"],
        "passes": report["passes"],
        "subgraphs": [{
            "operators": item["operators"],
            "custom_operators": item["custom_operators"],
            "internal_float_count": len(item["internal_float_operations"]),
            "internal_float_examples": item["internal_float_operations"][:8],
        } for item in report["subgraphs"]],
    }, indent=2))
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
