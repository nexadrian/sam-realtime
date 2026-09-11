from __future__ import annotations

import numpy as np

from ..backend import BackendSelection
from ..config import ModelEntry
from ..segmenter import Segmenter
from ..types import PromptEvidence, RuntimeSettings, SegmenterOutput, TrackingState
from .onnx_utils import require_paths, timed_call


class FastSamSegmenter(Segmenter):
    def __init__(self, model: ModelEntry, backend: BackendSelection | None = None) -> None:
        super().__init__(model, backend)
        (model_path,) = require_paths(model, "model_path")
        try:
            from ultralytics import FastSAM
        except ImportError as exc:
            raise RuntimeError(
                "FastSAM support requires Ultralytics. Install with the project fastsam extra, "
                "then place FastSAM-s.pt at the configured model_path."
            ) from exc

        self.model_path = model_path
        self.model_impl = FastSAM(str(model_path))

    def segment(
        self,
        frame_rgb: np.ndarray,
        settings: RuntimeSettings,
        state: TrackingState,
    ) -> SegmenterOutput:
        height, width = frame_rgb.shape[:2]
        point = (width // 2, height // 2)

        def run_model():
            return self.model_impl(
                frame_rgb,
                device=_ultralytics_device(self.backend),
                retina_masks=True,
                imgsz=settings.input_long_side,
                conf=0.4,
                iou=0.9,
                points=[[point[0], point[1]]],
                labels=[1],
                verbose=False,
            )

        results, inference_ms = timed_call(run_model)
        masks, scores = _extract_masks_and_scores(results, (height, width))
        return SegmenterOutput(
            masks=masks,
            scores=scores,
            prompt=PromptEvidence(prompt_type="center", positive_point=point),
            inference_ms=inference_ms,
        )


def _ultralytics_device(backend: BackendSelection | None) -> str:
    if backend and backend.name == "cuda":
        return "cuda"
    if backend and backend.name == "mps":
        return "mps"
    return "cpu"


def _extract_masks_and_scores(results, frame_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if not results:
        return np.zeros((0, *frame_shape), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    result = results[0]
    if getattr(result, "masks", None) is None or result.masks is None:
        return np.zeros((0, *frame_shape), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    mask_data = result.masks.data
    if hasattr(mask_data, "detach"):
        mask_data = mask_data.detach().cpu().numpy()
    else:
        mask_data = np.asarray(mask_data)

    if mask_data.ndim == 2:
        mask_data = mask_data[None]
    masks = mask_data.astype(np.float32)

    if getattr(result, "boxes", None) is not None and getattr(result.boxes, "conf", None) is not None:
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    else:
        scores = np.ones((masks.shape[0],), dtype=np.float32)

    if len(scores) != masks.shape[0]:
        scores = np.resize(scores, masks.shape[0]).astype(np.float32)
    return masks, scores
