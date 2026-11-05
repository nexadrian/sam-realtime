"""FastSAM LiteRT inference and camera application adapter."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ..backend import BackendSelection
from ..config import ModelEntry
from ..segmenter import Segmenter
from ..types import PromptEvidence, RuntimeSettings, SegmenterOutput, TrackingState
from .onnx_utils import require_paths


def crop_and_resize(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """Match the app's center-square crop and fixed-size camera input."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
        raise ValueError("Expected a BGR uint8 image with shape (height, width, 3)")
    height, width = frame_bgr.shape[:2]
    side = min(height, width)
    x0, y0 = (width - side) // 2, (height - side) // 2
    square = frame_bgr[y0 : y0 + side, x0 : x0 + side]
    if side == size:
        return np.ascontiguousarray(square)
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def model_input(frame_bgr: np.ndarray, input_detail: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return a prepared BGR display frame and the model's input tensor."""
    shape = tuple(int(value) for value in input_detail["shape"])
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError(f"Expected a fixed batch-one image input, got {shape}")
    if shape[1] == 3:
        size = shape[2]
        if shape[3] != size:
            raise ValueError(f"Expected a square model input, got {shape}")
        channels_first = True
    elif shape[3] == 3:
        size = shape[1]
        if shape[2] != size:
            raise ValueError(f"Expected a square model input, got {shape}")
        channels_first = False
    else:
        raise ValueError(f"Expected a 3-channel model input, got {shape}")

    prepared = crop_and_resize(frame_bgr, size)
    rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = rgb.transpose(2, 0, 1)[None] if channels_first else rgb[None]
    dtype = np.dtype(input_detail["dtype"])
    if np.issubdtype(dtype, np.integer):
        scales = input_detail["quantization_parameters"]["scales"]
        zero_points = input_detail["quantization_parameters"]["zero_points"]
        if len(scales) != 1 or float(scales[0]) <= 0:
            raise ValueError("Expected per-tensor quantization for the model input")
        limits = np.iinfo(dtype)
        tensor = np.clip(np.rint(tensor / float(scales[0]) + int(zero_points[0])), limits.min, limits.max)
    return prepared, np.ascontiguousarray(tensor.astype(dtype))


def _dequantize(tensor: np.ndarray, detail: dict) -> np.ndarray:
    if np.issubdtype(tensor.dtype, np.floating):
        return tensor.astype(np.float32, copy=False)
    params = detail["quantization_parameters"]
    scales = params["scales"]
    zero_points = params["zero_points"]
    if len(scales) != 1 or float(scales[0]) <= 0:
        raise ValueError(f"Expected per-tensor output quantization for {detail['name']}")
    return (tensor.astype(np.float32) - int(zero_points[0])) * float(scales[0])


def _decode_raw_head(box_logits: np.ndarray, score_logits: np.ndarray, mask_coefficients: np.ndarray,
                     size: int) -> np.ndarray:
    """Decode YOLO DFL distances and scores outside the integer LiteRT graph."""
    anchors = []
    strides = []
    for stride in (8, 16, 32):
        grid_size = size // stride
        yy, xx = np.meshgrid(np.arange(grid_size, dtype=np.float32) + 0.5,
                             np.arange(grid_size, dtype=np.float32) + 0.5, indexing="ij")
        anchors.append(np.stack((xx.ravel(), yy.ravel())))
        strides.append(np.full(grid_size * grid_size, stride / size, dtype=np.float32))
    anchor_points = np.concatenate(anchors, axis=1)
    scale = np.concatenate(strides)
    if box_logits.shape != (64, anchor_points.shape[1]):
        raise ValueError(f"Unexpected raw box shape {box_logits.shape} for input size {size}")
    distributions = box_logits.reshape(4, 16, -1)
    distributions = np.exp(distributions - distributions.max(axis=1, keepdims=True))
    distributions /= distributions.sum(axis=1, keepdims=True)
    distances = (distributions * np.arange(16, dtype=np.float32)[None, :, None]).sum(axis=1)
    xy1 = anchor_points - distances[:2]
    xy2 = anchor_points + distances[2:]
    boxes = np.concatenate(((xy1 + xy2) / 2, xy2 - xy1), axis=0) * scale
    scores = 1.0 / (1.0 + np.exp(-np.clip(score_logits, -80, 80)))
    return np.concatenate((boxes, scores, mask_coefficients), axis=0)


def _prediction_and_prototypes(outputs: list[np.ndarray], size: int) -> tuple[np.ndarray, np.ndarray]:
    if len(outputs) not in (2, 4):
        raise ValueError(f"Expected FastSAM raw or split outputs, got {len(outputs)} tensors")
    prototypes = next((value for value in outputs if value.ndim == 4), None)
    if prototypes is None or prototypes.shape[0] != 1:
        raise ValueError(f"Unexpected FastSAM output shapes: {[value.shape for value in outputs]}")
    proto = prototypes[0]
    if proto.shape[0] != 32 and proto.shape[-1] == 32:
        proto = proto.transpose(2, 0, 1)
    if proto.shape[0] != 32:
        raise ValueError(f"Expected 32 mask prototypes, got {proto.shape}")
    if len(outputs) == 2:
        proposals = next((value for value in outputs if value.ndim == 3), None)
        if proposals is None or proposals.shape[0] != 1:
            raise ValueError(f"Unexpected proposal output shapes: {[value.shape for value in outputs]}")
        pred = proposals[0]
    else:
        # Keeping these heads separate lets the quantizer assign each output
        # its own activation scale, especially for confidence scores.
        parts = {}
        for value in outputs:
            if value.ndim != 3:
                continue
            if value.shape[0] != 1:
                raise ValueError(f"Expected batch one, got {value.shape}")
            part = value[0]
            if part.shape[0] not in (1, 4, 32, 64) and part.shape[-1] in (1, 4, 32, 64):
                part = part.T
            parts[part.shape[0]] = part
        if len({part.shape[1] for part in parts.values()}) != 1:
            raise ValueError(f"Unexpected split proposal shapes: {[value.shape for value in outputs]}")
        if set(parts) == {1, 64, 32}:
            pred = _decode_raw_head(parts[64], parts[1], parts[32], size)
        elif set(parts) == {1, 4, 32}:
            pred = np.concatenate((parts[4], parts[1], parts[32]), axis=0)
        else:
            raise ValueError(f"Unexpected split proposal shapes: {[value.shape for value in outputs]}")
    if pred.shape[0] != 37 and pred.shape[-1] == 37:
        pred = pred.T
    if pred.shape[0] != 37:
        raise ValueError(f"Expected 4 box, 1 score, 32 mask channels, got {pred.shape}")
    return pred, proto


def _greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_masks: int) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    while order.size and len(keep) < max_masks:
        index = int(order[0])
        keep.append(index)
        remaining = order[1:]
        if not remaining.size:
            break
        left = np.maximum(boxes[index, 0], boxes[remaining, 0])
        top = np.maximum(boxes[index, 1], boxes[remaining, 1])
        right = np.minimum(boxes[index, 2], boxes[remaining, 2])
        bottom = np.minimum(boxes[index, 3], boxes[remaining, 3])
        intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
        union = areas[index] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= iou_threshold]
    return keep


def decode_masks_with_scores(
    outputs: list[np.ndarray],
    size: int,
    *,
    confidence: float = 0.4,
    iou: float = 0.9,
    mask_threshold: float = 0.5,
    max_masks: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode FastSAM outputs to center-point masks and proposal scores.

    Ultralytics' LiteRT exporter normalizes xywh boxes to [0, 1]. Mask
    prototypes remain logits; comparing them with logit(threshold) avoids a
    separate sigmoid and matches the 0.5 default at a zero-logit boundary.
    """
    if not (0 <= confidence <= 1 and 0 <= iou <= 1 and 0 <= mask_threshold <= 1):
        raise ValueError("Thresholds must be between zero and one")
    pred, proto = _prediction_and_prototypes(outputs, size)
    scores = pred[4]
    selected = np.flatnonzero(scores > confidence)
    if selected.size == 0:
        return np.empty((0, size, size), dtype=np.uint8), np.empty((0,), dtype=np.float32)

    xywh = pred[:4, selected].T * size
    boxes = np.empty_like(xywh)
    boxes[:, :2] = xywh[:, :2] - xywh[:, 2:] / 2
    boxes[:, 2:] = xywh[:, :2] + xywh[:, 2:] / 2
    chosen = _greedy_nms(boxes, scores[selected], iou, max_masks)
    if not chosen:
        return np.empty((0, size, size), dtype=np.uint8), np.empty((0,), dtype=np.float32)

    if mask_threshold <= 0:
        logit_threshold = -np.inf
    elif mask_threshold >= 1:
        logit_threshold = np.inf
    else:
        logit_threshold = float(np.log(mask_threshold / (1.0 - mask_threshold)))
    center = size // 2
    yy, xx = np.ogrid[:size, :size]
    masks = []
    mask_scores = []
    for index in chosen:
        coeff = pred[5:, selected[index]]
        logits = coeff @ proto.reshape(32, -1)
        logits = cv2.resize(logits.reshape(proto.shape[1:]), (size, size), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = boxes[index]
        mask = ((logits > logit_threshold) & (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)).astype(np.uint8)
        if mask[center, center]:
            masks.append(mask)
            mask_scores.append(scores[selected[index]])
    if not masks:
        return np.empty((0, size, size), dtype=np.uint8), np.empty((0,), dtype=np.float32)
    return np.stack(masks), np.asarray(mask_scores, dtype=np.float32)


def decode_masks(outputs: list[np.ndarray], size: int, **thresholds) -> np.ndarray:
    """Return only binary masks for callers that do not need scores."""
    return decode_masks_with_scores(outputs, size, **thresholds)[0]


class FastSamTFLite:
    def __init__(self, model_path: str | Path, *, threads: int | None = None) -> None:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            raise RuntimeError("Install ai-edge-litert to run FastSAM TFLite models") from exc
        kwargs = {"model_path": str(model_path)}
        if threads is not None:
            kwargs["num_threads"] = threads
        self.interpreter = Interpreter(**kwargs)
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        if len(inputs) != 1:
            raise ValueError(f"Expected one model input, got {len(inputs)}")
        self.input_detail = inputs[0]
        self.output_details = self.interpreter.get_output_details()

    def infer_prepared_with_scores(self, frame_bgr: np.ndarray, **thresholds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prepared, tensor = model_input(frame_bgr, self.input_detail)
        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        self.interpreter.invoke()
        outputs = [
            _dequantize(self.interpreter.get_tensor(detail["index"]), detail)
            for detail in self.output_details
        ]
        masks, scores = decode_masks_with_scores(outputs, prepared.shape[0], **thresholds)
        return prepared, masks, scores

    def infer_prepared(self, frame_bgr: np.ndarray, **thresholds) -> tuple[np.ndarray, np.ndarray]:
        prepared, masks, _ = self.infer_prepared_with_scores(frame_bgr, **thresholds)
        return prepared, masks

    def infer(self, frame_bgr: np.ndarray, **thresholds) -> np.ndarray:
        """Return binary center-point masks shaped (N, S, S)."""
        _, masks = self.infer_prepared(frame_bgr, **thresholds)
        return masks


class FastSamTFLiteSegmenter(Segmenter):
    """Run a fixed-size FastSAM LiteRT model inside the camera application."""

    def __init__(self, model: ModelEntry, backend: BackendSelection | None = None) -> None:
        super().__init__(model, backend)
        (model_path,) = require_paths(model, "model_path")
        self.runtime = FastSamTFLite(model_path)

    def segment(
        self,
        frame_rgb: np.ndarray,
        settings: RuntimeSettings,
        state: TrackingState,
    ) -> SegmenterOutput:
        height, width = frame_rgb.shape[:2]
        point = (width // 2, height // 2)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        start = time.perf_counter()
        _, masks, scores = self.runtime.infer_prepared_with_scores(
            frame_bgr,
            confidence=settings.score_threshold,
            iou=0.9,
            mask_threshold=settings.mask_threshold,
        )
        inference_ms = (time.perf_counter() - start) * 1000.0

        # The app's resolution trackbar may differ from the model's fixed
        # input size; masks must match the displayed frame for selection.
        if masks.shape[1:] != (height, width):
            masks = (
                np.stack([cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) for mask in masks])
                if len(masks) else np.empty((0, height, width), dtype=np.uint8)
            )
        return SegmenterOutput(
            masks=masks,
            scores=scores,
            prompt=PromptEvidence(prompt_type="center", positive_point=point),
            inference_ms=inference_ms,
        )
