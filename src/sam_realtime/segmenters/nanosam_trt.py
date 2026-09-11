from __future__ import annotations

import numpy as np

from ..backend import BackendSelection
from ..config import ModelEntry
from ..segmenter import Segmenter
from ..types import PromptEvidence, RuntimeSettings, SegmenterOutput, TrackingState
from .onnx_utils import require_paths, timed_call


class NanoSamTrtSegmenter(Segmenter):
    def __init__(self, model: ModelEntry, backend: BackendSelection | None = None) -> None:
        super().__init__(model, backend)
        image_encoder, mask_decoder = require_paths(model, "encoder_engine_path", "decoder_engine_path")
        try:
            from nanosam.utils.predictor import Predictor
        except ImportError as exc:
            raise RuntimeError(
                "NanoSAM support requires the upstream nanosam package plus TensorRT. "
                "Build target-specific .engine files using https://github.com/NVIDIA-AI-IOT/nanosam."
            ) from exc

        self.predictor = Predictor(image_encoder=str(image_encoder), mask_decoder=str(mask_decoder))

    def segment(
        self,
        frame_rgb: np.ndarray,
        settings: RuntimeSettings,
        state: TrackingState,
    ) -> SegmenterOutput:
        from PIL import Image

        height, width = frame_rgb.shape[:2]
        point = (width // 2, height // 2)

        def run_model():
            self.predictor.set_image(Image.fromarray(frame_rgb))
            mask, scores, _logits = self.predictor.predict(
                np.array([point], dtype=np.float32),
                np.array([1], dtype=np.int32),
            )
            return mask, scores

        (mask, scores), inference_ms = timed_call(run_model)
        masks = np.asarray(mask, dtype=np.float32)
        if masks.ndim == 2:
            masks = masks[None]
        return SegmenterOutput(
            masks=masks,
            scores=np.asarray(scores, dtype=np.float32).reshape(-1),
            prompt=PromptEvidence(prompt_type="center", positive_point=point),
            inference_ms=inference_ms,
        )
