from __future__ import annotations

import numpy as np

from ..backend import BackendSelection
from ..config import ModelEntry
from ..segmenter import Segmenter
from ..types import PromptEvidence, RuntimeSettings, SegmenterOutput, TrackingState
from .onnx_utils import require_paths, timed_call


class Sam2TorchSegmenter(Segmenter):
    def __init__(self, model: ModelEntry, backend: BackendSelection | None = None) -> None:
        super().__init__(model, backend)
        checkpoint_path, config_path = require_paths(model, "model_path", "config_path")
        try:
            import torch
            from hydra import initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM2 support requires the optional upstream sam2 package. "
                "Install it from https://github.com/facebookresearch/sam2, then retry --model sam2_torch."
            ) from exc

        device = _torch_device(torch, backend)
        # SAM 2 initializes Hydra with its package directory, so an absolute
        # config path from our registry cannot be passed directly to build_sam2.
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(config_path.parent), version_base="1.2"):
            sam2_model = build_sam2(config_path.name, str(checkpoint_path), device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def segment(
        self,
        frame_rgb: np.ndarray,
        settings: RuntimeSettings,
        state: TrackingState,
    ) -> SegmenterOutput:
        height, width = frame_rgb.shape[:2]
        point = (width // 2, height // 2)

        def run_model():
            self.predictor.set_image(frame_rgb)
            return self.predictor.predict(
                point_coords=np.array([point], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )

        (masks, scores, _logits), inference_ms = timed_call(run_model)
        return SegmenterOutput(
            masks=np.asarray(masks, dtype=np.float32),
            scores=np.asarray(scores, dtype=np.float32),
            prompt=PromptEvidence(prompt_type="center", positive_point=point),
            inference_ms=inference_ms,
        )


def _torch_device(torch, backend: BackendSelection | None):
    if backend and backend.name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if backend and backend.name == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
    return torch.device("cpu")
