from __future__ import annotations

import numpy as np

from ..backend import BackendSelection
from ..config import ModelEntry
from ..segmenter import Segmenter
from ..types import PromptEvidence, RuntimeSettings, SegmenterOutput, TrackingState
from .onnx_utils import make_session, point_to_model, prepare_square_image, require_paths, restore_masks, sigmoid, timed_call


class SamOnnxSegmenter(Segmenter):
    def __init__(self, model: ModelEntry, backend: BackendSelection | None = None) -> None:
        super().__init__(model, backend)
        encoder_path, decoder_path = require_paths(model, "encoder_path", "decoder_path")
        self.encoder = make_session(encoder_path, backend)
        self.decoder = make_session(decoder_path, backend)
        self.encoder_input_name = self.encoder.get_inputs()[0].name
        self.decoder_input_names = {item.name for item in self.decoder.get_inputs()}

    def segment(
        self,
        frame_rgb: np.ndarray,
        settings: RuntimeSettings,
        state: TrackingState,
    ) -> SegmenterOutput:
        import cv2

        prepared = prepare_square_image(
            cv2,
            frame_rgb,
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        )
        height, width = frame_rgb.shape[:2]
        point = (width // 2, height // 2)
        model_point = point_to_model(point, prepared)

        def run_model():
            (embeddings,) = self.encoder.run(None, {self.encoder_input_name: prepared.input_tensor})
            decoder_inputs = {
                "image_embeddings": embeddings,
                "point_coords": np.array([[model_point]], dtype=np.float32),
                "point_labels": np.array([[1]], dtype=np.float32),
            }
            if "mask_input" in self.decoder_input_names:
                decoder_inputs["mask_input"] = np.zeros((1, 1, 256, 256), dtype=np.float32)
            if "has_mask_input" in self.decoder_input_names:
                decoder_inputs["has_mask_input"] = np.array([0], dtype=np.float32)
            if "orig_im_size" in self.decoder_input_names:
                decoder_inputs["orig_im_size"] = np.array([height, width], dtype=np.float32)
            outputs = self.decoder.run(None, decoder_inputs)
            return outputs

        outputs, inference_ms = timed_call(run_model)
        masks = np.asarray(outputs[0], dtype=np.float32)
        scores = np.asarray(outputs[1], dtype=np.float32) if len(outputs) > 1 else np.ones((1,), dtype=np.float32)

        while masks.ndim > 3:
            masks = masks[0]
        if masks.ndim == 2:
            masks = masks[None]
        masks = sigmoid(masks)
        if masks.shape[-2:] != (height, width):
            masks = restore_masks(cv2, masks, prepared)
        scores = scores.reshape(-1).astype(np.float32)

        return SegmenterOutput(
            masks=masks,
            scores=scores,
            prompt=PromptEvidence(prompt_type="center", positive_point=point),
            inference_ms=inference_ms,
        )
