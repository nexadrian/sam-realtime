from __future__ import annotations

import time

import numpy as np

from ..backend import BackendSelection
from ..config import ModelEntry
from ..segmenter import Segmenter
from ..types import PromptEvidence, RuntimeSettings, SegmenterOutput, TrackingState


class DummySegmenter(Segmenter):
    def __init__(self, model: ModelEntry, backend: BackendSelection | None = None) -> None:
        super().__init__(model, backend)

    def segment(
        self,
        frame_rgb: np.ndarray,
        settings: RuntimeSettings,
        state: TrackingState,
    ) -> SegmenterOutput:
        start = time.perf_counter()
        height, width = frame_rgb.shape[:2]
        center = (width // 2, height // 2)
        radius = max(12, min(width, height) // 5)

        yy, xx = np.ogrid[:height, :width]
        distance_sq = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        mask = distance_sq <= radius**2
        inference_ms = (time.perf_counter() - start) * 1000.0

        return SegmenterOutput(
            masks=np.expand_dims(mask.astype(np.float32), axis=0),
            scores=np.array([1.0], dtype=np.float32),
            prompt=PromptEvidence(prompt_type="center", positive_point=center),
            inference_ms=inference_ms,
        )
