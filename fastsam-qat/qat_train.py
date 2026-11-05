"""Fine-tune FastSAM-s with PT2E fake quantization and float-teacher distillation.

This trains in PyTorch, saves ordinary FastSAM weights, and leaves TFLite
export to the existing full-integer post-training converter. The direct PT2E
TFLite conversion currently leaves float islands, so exported models must
still pass audit_graph.py and held-out mask evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchao.quantization.pt2e import quantize_pt2e
from litert_torch.quantize import pt2e_quantizer
from ultralytics import FastSAM
from ultralytics.nn.modules.head import Detect

from qat_probe import RawFastSAM


def prepare_model(checkpoint: Path, size: int) -> RawFastSAM:
    model = FastSAM(str(checkpoint)).model.float().eval()
    model.fuse()
    model.requires_grad_(True)
    for layer in model.modules():
        if isinstance(layer, Detect):
            layer.export = True
            layer.format = "litert"
    return RawFastSAM(model, size).eval()


def image_tensor(path: Path, size: int) -> torch.Tensor:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Could not read {path}")
    height, width = frame.shape[:2]
    side = min(height, width)
    square = frame[(height - side) // 2 : (height + side) // 2,
                   (width - side) // 2 : (width + side) // 2]
    square = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)


def distillation_loss(student: tuple[torch.Tensor, ...], teacher: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, dict[str, float]]:
    boxes_s, scores_s, coeff_s, proto_s = student
    boxes_t, scores_t, coeff_t, proto_t = teacher
    # A 0.4 confidence threshold controls the downstream mask count. The
    # extra weight in its neighborhood aims at the observed failure mode.
    near_threshold = (scores_t - 0.4).abs() < 0.2
    score_weight = 1.0 + 20.0 * near_threshold.float() + 4.0 * (scores_t > 0.4).float()
    score = ((scores_s - scores_t).square() * score_weight).mean()
    box_weight = 1.0 + 4.0 * (scores_t > 0.2).float()
    box = (F.smooth_l1_loss(boxes_s, boxes_t, reduction="none") * box_weight).mean()
    coeff = F.smooth_l1_loss(coeff_s, coeff_t)
    proto = F.smooth_l1_loss(proto_s, proto_t)
    loss = 20.0 * score + 2.0 * box + 0.1 * coeff + 0.1 * proto
    parts = {"score": float(score.detach()), "box": float(box.detach()),
             "coeff": float(coeff.detach()), "proto": float(proto.detach()), "total": float(loss.detach())}
    return loss, parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--camera-train", type=int, default=250)
    parser.add_argument("--coco-train", type=int, default=100)
    parser.add_argument("--trainable", choices=("full", "score_head"), default="full")
    args = parser.parse_args()
    if args.batch < 1 or args.epochs < 1:
        parser.error("Batch and epochs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    camera = sorted(args.camera.glob("*.jpg"))
    coco = sorted(args.coco.glob("*.jpg"))
    if len(camera) < args.camera_train + 20 or len(coco) < args.coco_train + 20:
        parser.error(f"Not enough images: camera={len(camera)}, COCO={len(coco)}")
    training = camera[: args.camera_train] + coco[: args.coco_train]
    selection = camera[args.camera_train :] + coco[args.coco_train :]
    (args.output / "data_split.json").write_text(json.dumps({
        "train": [str(path) for path in training],
        "selection": [str(path) for path in selection],
        "holdout_note": "Never train or select on original holdout or fresh camera frames",
    }, indent=2) + "\n")

    teacher = prepare_model(args.checkpoint, args.size)
    teacher.requires_grad_(False)
    student = prepare_model(args.checkpoint, args.size)
    example = torch.zeros(args.batch, 3, args.size, args.size)
    with torch.no_grad():
        student(example)
        teacher(example)
    exported = torch.export.export(student, (example,)).module()
    # All BN layers were fused before export. This skips a LiteRT Torch 0.9.4
    # Conv+BN QAT pattern builder that is incompatible with torchao 0.18.
    assert not any(isinstance(layer, torch.nn.BatchNorm2d) for layer in student.modules())
    pt2e_quantizer.PT2EQuantizer.STATIC_QAT_ONLY_OPS = []
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_qat=True, is_per_channel=True)
    )
    prepared = quantize_pt2e.prepare_qat_pt2e(exported, quantizer).cuda()
    teacher = teacher.cuda()
    fake_quant_count = sum("FakeQuant" in type(layer).__name__ for layer in prepared.modules())
    if fake_quant_count < 50:
        raise RuntimeError(f"Only {fake_quant_count} fake-quant modules were inserted")
    if args.trainable == "score_head":
        for name, parameter in prepared.named_parameters():
            parameter.requires_grad_(name.startswith("model.model.22.cv3."))
    trainable_params = [parameter for parameter in prepared.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"No trainable parameters for {args.trainable}")
    print(f"QAT ready: {fake_quant_count} fake-quant modules, {len(training)} train, {len(selection)} selection images, "
          f"{sum(p.numel() for p in trainable_params)} trainable parameters ({args.trainable})", flush=True)
    original_keys = list(student.state_dict())
    if not set(original_keys).issubset(prepared.state_dict()):
        raise RuntimeError("Cannot transfer trained weights back to float FastSAM")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        rng = random.Random(args.seed + epoch)
        order = training.copy()
        rng.shuffle(order)
        metrics = []
        for offset in range(0, len(order) - args.batch + 1, args.batch):
            paths = order[offset : offset + args.batch]
            images = torch.stack([image_tensor(path, args.size) for path in paths]).cuda(non_blocking=True)
            with torch.no_grad():
                reference = teacher(images)
            optimizer.zero_grad(set_to_none=True)
            prediction = prepared(images)
            loss, parts = distillation_loss(prediction, reference)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            metrics.append(parts)
            if len(metrics) % 20 == 0:
                print(f"epoch {epoch} step {len(metrics)}/{len(order)//args.batch} loss={parts['total']:.5f}", flush=True)
        means = {key: sum(item[key] for item in metrics) / len(metrics) for key in metrics[0]}
        means.update({"epoch": epoch, "steps": len(metrics), "elapsed_seconds": time.monotonic() - started})
        history.append(means)
        trained = prepared.state_dict()
        weights = {key: trained[key].detach().cpu().clone() for key in original_keys}
        checkpoint_path = args.output / f"weights_epoch{epoch}.pt"
        torch.save({"state_dict": weights, "epoch": epoch, "training": means,
                    "size": args.size, "batch": args.batch, "fake_quant_modules": fake_quant_count,
                    "trainable": args.trainable}, checkpoint_path)
        (args.output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"saved {checkpoint_path} metrics={means}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
