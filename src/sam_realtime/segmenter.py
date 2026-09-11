from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .backend import BackendSelection
from .config import ModelEntry
from .types import RuntimeSettings, SegmenterOutput, TrackingState


class Segmenter(ABC):
    def __init__(self, model: ModelEntry, backend: BackendSelection | None = None) -> None:
        self.model = model
        self.backend = backend

    @abstractmethod
    def segment(
        self,
        frame_rgb: np.ndarray,
        settings: RuntimeSettings,
        state: TrackingState,
    ) -> SegmenterOutput:
        """Return candidate masks and scores for one RGB frame."""


def build_segmenter(model: ModelEntry, backend: BackendSelection | None = None) -> Segmenter:
    if model.type == "dummy":
        from .segmenters.dummy import DummySegmenter

        return DummySegmenter(model, backend)
    if model.type == "efficient_sam_onnx":
        from .segmenters.efficient_sam_onnx import EfficientSamOnnxSegmenter

        return EfficientSamOnnxSegmenter(model, backend)
    if model.type == "mobile_sam_onnx":
        from .segmenters.mobile_sam_onnx import MobileSamOnnxSegmenter

        return MobileSamOnnxSegmenter(model, backend)
    if model.type == "sam_onnx":
        from .segmenters.sam_onnx import SamOnnxSegmenter

        return SamOnnxSegmenter(model, backend)
    if model.type == "sam2_torch":
        from .segmenters.sam2_torch import Sam2TorchSegmenter

        return Sam2TorchSegmenter(model, backend)
    if model.type == "nanosam_trt":
        from .segmenters.nanosam_trt import NanoSamTrtSegmenter

        return NanoSamTrtSegmenter(model, backend)
    if model.type == "fastsam":
        from .segmenters.fastsam import FastSamSegmenter

        return FastSamSegmenter(model, backend)
    raise ValueError(f"Unsupported model type {model.type!r}")
