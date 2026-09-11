from __future__ import annotations

import textwrap

import numpy as np

from ..config import ModelEntry
from ..segmenter import Segmenter
from ..types import RuntimeSettings, SegmenterOutput, TrackingState


class PlannedSegmenter(Segmenter):
    def __init__(self, model: ModelEntry) -> None:
        super().__init__(model)
        paths = "\n".join(f"  - {path}" for path in model.paths) or "  - no model path configured"
        raise NotImplementedError(
            textwrap.dedent(
                f"""
                The {model.name!r} model is registered but its segmenter adapter is not implemented yet.

                Type: {model.type}
                Backend: {model.backend}
                Expected path(s):
                {paths}

                Run with --list-models to inspect available models, or run with --model dummy.
                """
            ).strip()
        )

    def segment(
        self,
        frame_rgb: np.ndarray,
        settings: RuntimeSettings,
        state: TrackingState,
    ) -> SegmenterOutput:
        raise NotImplementedError
