from __future__ import annotations

import logging
from pathlib import Path
import time

import numpy as np

from .backend import BackendSelection
from .config import ModelEntry
from .mask import choose_mask, mask_to_box
from .render import render_frame, save_outputs
from .segmenter import build_segmenter
from .types import FrameTimings, RuntimeSettings, SelectedMask, TrackingState, ViewMode

LOGGER = logging.getLogger(__name__)
WINDOW_NAME = "sam-realtime"


def run_app(
    *,
    model: ModelEntry,
    backend: BackendSelection,
    camera_index: int,
    settings: RuntimeSettings,
    output_dir: Path,
) -> int:
    import cv2

    segmenter = build_segmenter(model, backend)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    _create_trackbars(cv2, settings)

    state = TrackingState()
    view_mode = ViewMode.OVERLAY
    paused = False
    last_frame_bgr: np.ndarray | None = None
    selected: SelectedMask | None = None
    prompt = None
    last_loop_start = time.perf_counter()
    last_render_ms = 0.0

    LOGGER.info("Keyboard controls: q quit, r reset, v view, p pause, s save")

    try:
        while True:
            loop_start = time.perf_counter()
            settings = _read_trackbars(cv2, settings)

            capture_start = time.perf_counter()
            if not paused or last_frame_bgr is None:
                ok, frame_bgr = capture.read()
                if not ok:
                    LOGGER.warning("Camera frame capture failed")
                    break
                last_frame_bgr = frame_bgr
            capture_ms = (time.perf_counter() - capture_start) * 1000.0

            frame_rgb = cv2.cvtColor(last_frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = _center_crop_square(frame_rgb)
            frame_rgb = _resize_long_side(cv2, frame_rgb, settings.input_long_side)

            output = segmenter.segment(frame_rgb, settings, state)
            selected = choose_mask(
                output.masks,
                output.scores,
                output.prompt,
                mask_threshold=settings.mask_threshold,
                score_threshold=settings.score_threshold,
            )
            prompt = output.prompt
            tracking_lost = selected is None

            if selected is None:
                state.lost_frames += 1
                state.last_mask = None
                state.last_box = None
            else:
                state.lost_frames = 0
                state.last_mask = selected.mask
                state.last_box = mask_to_box(selected.mask)

            render_start = time.perf_counter()
            loop_duration = loop_start - last_loop_start
            last_loop_start = loop_start
            timings = FrameTimings(
                capture_ms=capture_ms,
                inference_ms=output.inference_ms,
                render_ms=last_render_ms,
                loop_fps=(1.0 / loop_duration) if loop_duration > 0 else 0.0,
            )
            rendered = render_frame(
                cv2,
                frame_rgb,
                selected,
                prompt,
                view_mode,
                timings,
                model_name=model.name,
                backend_name=backend.name,
                overlay_alpha=settings.overlay_alpha,
                tracking_lost=tracking_lost,
            )
            last_render_ms = (time.perf_counter() - render_start) * 1000.0

            cv2.imshow(WINDOW_NAME, rendered)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                state.reset()
                LOGGER.info("Tracking reset")
            elif key == ord("v"):
                view_mode = ViewMode.cycle(view_mode)
            elif key == ord("p"):
                paused = not paused
                LOGGER.info("Paused: %s", paused)
            elif key == ord("s"):
                image_path, mask_path = save_outputs(cv2, output_dir, rendered, selected.mask if selected else None)
                LOGGER.info("Saved screenshot to %s%s", image_path, f" and mask to {mask_path}" if mask_path else "")

            state.frame_index += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


def _create_trackbars(cv2_module, settings: RuntimeSettings) -> None:
    cv2_module.createTrackbar("mask threshold", WINDOW_NAME, int(settings.mask_threshold * 100), 100, lambda _value: None)
    cv2_module.createTrackbar("overlay alpha", WINDOW_NAME, int(settings.overlay_alpha * 100), 100, lambda _value: None)
    cv2_module.createTrackbar("input long side", WINDOW_NAME, settings.input_long_side, 1536, lambda _value: None)
    cv2_module.createTrackbar("score threshold", WINDOW_NAME, int(settings.score_threshold * 100), 100, lambda _value: None)
    cv2_module.createTrackbar("refresh interval", WINDOW_NAME, settings.refresh_interval, 120, lambda _value: None)
    cv2_module.createTrackbar("smoothing", WINDOW_NAME, int(settings.smoothing * 100), 100, lambda _value: None)
    cv2_module.createTrackbar("failure grace", WINDOW_NAME, settings.failure_grace_frames, 60, lambda _value: None)


def _read_trackbars(cv2_module, previous: RuntimeSettings) -> RuntimeSettings:
    return RuntimeSettings(
        mask_threshold=cv2_module.getTrackbarPos("mask threshold", WINDOW_NAME) / 100.0,
        overlay_alpha=cv2_module.getTrackbarPos("overlay alpha", WINDOW_NAME) / 100.0,
        input_long_side=max(64, cv2_module.getTrackbarPos("input long side", WINDOW_NAME)),
        score_threshold=cv2_module.getTrackbarPos("score threshold", WINDOW_NAME) / 100.0,
        refresh_interval=max(1, cv2_module.getTrackbarPos("refresh interval", WINDOW_NAME)),
        smoothing=cv2_module.getTrackbarPos("smoothing", WINDOW_NAME) / 100.0,
        failure_grace_frames=max(0, cv2_module.getTrackbarPos("failure grace", WINDOW_NAME)),
    )


def _center_crop_square(frame_rgb: np.ndarray) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    side = min(height, width)
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    return frame_rgb[y0 : y0 + side, x0 : x0 + side]


def _resize_long_side(cv2_module, frame_rgb: np.ndarray, long_side: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    current_long_side = max(height, width)
    if current_long_side == long_side:
        return frame_rgb
    scale = long_side / current_long_side
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2_module.resize(frame_rgb, new_size, interpolation=cv2_module.INTER_AREA)
