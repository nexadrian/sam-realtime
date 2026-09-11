from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class ViewMode(Enum):
    OVERLAY = "overlay"
    ISOLATED = "isolated"
    MASK = "mask"
    SIDE_BY_SIDE = "side-by-side"

    @classmethod
    def cycle(cls, current: "ViewMode") -> "ViewMode":
        modes = list(cls)
        return modes[(modes.index(current) + 1) % len(modes)]


@dataclass(frozen=True)
class RuntimeSettings:
    input_long_side: int = 512
    mask_threshold: float = 0.5
    score_threshold: float = 0.5
    overlay_alpha: float = 0.45
    refresh_interval: int = 1
    smoothing: float = 0.0
    failure_grace_frames: int = 3


@dataclass(frozen=True)
class PromptEvidence:
    prompt_type: Literal["center", "previous-mask", "previous-box"]
    positive_point: tuple[int, int] | None = None
    prior_box: tuple[int, int, int, int] | None = None
    prior_mask: object | None = None


@dataclass(frozen=True)
class SegmenterOutput:
    masks: object
    scores: object
    prompt: PromptEvidence
    inference_ms: float


@dataclass(frozen=True)
class SelectedMask:
    mask: object
    score: float
    area: int


@dataclass
class TrackingState:
    last_mask: object | None = None
    last_box: tuple[int, int, int, int] | None = None
    lost_frames: int = 0
    frame_index: int = 0

    def reset(self) -> None:
        self.last_mask = None
        self.last_box = None
        self.lost_frames = 0
        self.frame_index = 0


@dataclass(frozen=True)
class FrameTimings:
    capture_ms: float = 0.0
    inference_ms: float = 0.0
    render_ms: float = 0.0
    loop_fps: float = 0.0
