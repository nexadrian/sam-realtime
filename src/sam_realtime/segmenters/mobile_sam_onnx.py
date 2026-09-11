from __future__ import annotations

import numpy as np

from ..backend import BackendSelection
from ..config import ModelEntry
from ..segmenter import Segmenter
from ..types import PromptEvidence, RuntimeSettings, SegmenterOutput, TrackingState
from .onnx_utils import make_session, point_to_model, prepare_square_image, require_paths, restore_masks, sigmoid, timed_call


class MobileSamOnnxSegmenter(Segmenter):
    def __init__(self, model: ModelEntry, backend: BackendSelection | None = None) -> None:
        super().__init__(model, backend)
        encoder_path, decoder_path = require_paths(model, "encoder_path", "decoder_path")
        self.encoder = make_session(encoder_path, backend)
        self.decoder = make_session(decoder_path, backend)

    def segment(
        self,
        frame_rgb: np.ndarray,
        settings: RuntimeSettings,
        state: TrackingState,
    ) -> SegmenterOutput:
        import cv2

        prepared = prepare_square_image(cv2, frame_rgb)
        height, width = frame_rgb.shape[:2]
        point = (width // 2, height // 2)
        model_point = point_to_model(point, prepared)

        def run_model():
            (embeddings,) = self.encoder.run(None, {"image": prepared.input_tensor})
            masks, scores = self.decoder.run(
                None,
                {
                    "image_embeddings": embeddings,
                    "point_coords": np.array([[model_point]], dtype=np.float32),
                    "point_labels": np.array([[1]], dtype=np.float32),
                },
            )
            return masks, scores

        (masks, scores), inference_ms = timed_call(run_model)
        masks = sigmoid(np.asarray(masks[0], dtype=np.float32))
        scores = np.asarray(scores[0], dtype=np.float32)
        restored_masks = restore_masks(cv2, masks, prepared)

        return SegmenterOutput(
            masks=restored_masks,
            scores=scores,
            prompt=PromptEvidence(prompt_type="center", positive_point=point),
            inference_ms=inference_ms,
        )
