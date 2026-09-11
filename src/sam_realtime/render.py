from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import FrameTimings, PromptEvidence, SelectedMask, ViewMode


MASK_COLOR_RGB = np.array([0, 220, 140], dtype=np.uint8)
TEXT_COLOR = (245, 245, 245)
WARNING_COLOR = (80, 220, 255)
PROMPT_COLOR = (40, 230, 255)


def render_frame(
    cv2_module,
    frame_rgb: np.ndarray,
    selected: SelectedMask | None,
    prompt: PromptEvidence | None,
    view_mode: ViewMode,
    timings: FrameTimings,
    *,
    model_name: str,
    backend_name: str,
    overlay_alpha: float,
    tracking_lost: bool,
) -> np.ndarray:
    if selected is None:
        mask = np.zeros(frame_rgb.shape[:2], dtype=bool)
        score = None
        area = 0
    else:
        mask = selected.mask
        score = selected.score
        area = selected.area

    if view_mode == ViewMode.OVERLAY:
        canvas_rgb = _overlay(frame_rgb, mask, overlay_alpha)
    elif view_mode == ViewMode.ISOLATED:
        canvas_rgb = np.where(mask[..., None], frame_rgb, 0)
    elif view_mode == ViewMode.MASK:
        canvas_rgb = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    else:
        overlay = _overlay(frame_rgb, mask, overlay_alpha)
        isolated = np.where(mask[..., None], frame_rgb, 0)
        canvas_rgb = np.concatenate([overlay, isolated], axis=1)

    canvas_bgr = cv2_module.cvtColor(canvas_rgb, cv2_module.COLOR_RGB2BGR)
    _draw_metrics(
        cv2_module,
        canvas_bgr,
        timings,
        model_name=model_name,
        backend_name=backend_name,
        score=score,
        area=area,
        view_mode=view_mode,
        tracking_lost=tracking_lost,
    )
    if prompt is not None:
        _draw_prompt(cv2_module, canvas_bgr, prompt)
    return canvas_bgr


def save_outputs(cv2_module, output_dir: Path, rendered_bgr: np.ndarray, mask: np.ndarray | None) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        image_path = output_dir / f"capture_{index:04d}.png"
        mask_path = output_dir / f"mask_{index:04d}.png"
        if not image_path.exists() and not mask_path.exists():
            break
        index += 1

    cv2_module.imwrite(str(image_path), rendered_bgr)
    if mask is None:
        return image_path, None
    cv2_module.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
    return image_path, mask_path


def _overlay(frame_rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    alpha = min(max(alpha, 0.0), 1.0)
    tinted = frame_rgb.copy()
    tinted[mask] = (
        frame_rgb[mask].astype(np.float32) * (1.0 - alpha)
        + MASK_COLOR_RGB.astype(np.float32) * alpha
    ).astype(np.uint8)
    return tinted


def _draw_metrics(
    cv2_module,
    canvas_bgr: np.ndarray,
    timings: FrameTimings,
    *,
    model_name: str,
    backend_name: str,
    score: float | None,
    area: int,
    view_mode: ViewMode,
    tracking_lost: bool,
) -> None:
    score_text = "none" if score is None else f"{score:.2f}"
    lines = [
        f"FPS {timings.loop_fps:5.1f}",
        f"cap {timings.capture_ms:5.1f}ms  infer {timings.inference_ms:5.1f}ms  render {timings.render_ms:5.1f}ms",
        f"{model_name} / {backend_name}",
        f"score {score_text}  area {area}  view {view_mode.value}",
    ]
    if tracking_lost:
        lines.append("tracking lost")

    x, y = 12, 24
    for line in lines:
        cv2_module.putText(canvas_bgr, line, (x, y), cv2_module.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2_module.LINE_AA)
        cv2_module.putText(
            canvas_bgr,
            line,
            (x, y),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.55,
            WARNING_COLOR if tracking_lost and line == "tracking lost" else TEXT_COLOR,
            1,
            cv2_module.LINE_AA,
        )
        y += 22


def _draw_prompt(cv2_module, canvas_bgr: np.ndarray, prompt: PromptEvidence) -> None:
    if prompt.positive_point is not None:
        cv2_module.drawMarker(
            canvas_bgr,
            prompt.positive_point,
            PROMPT_COLOR,
            markerType=cv2_module.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    if prompt.prior_box is not None:
        x0, y0, x1, y1 = prompt.prior_box
        cv2_module.rectangle(canvas_bgr, (x0, y0), (x1, y1), PROMPT_COLOR, 2)

    if prompt.prior_mask is not None:
        contours, _ = cv2_module.findContours(prompt.prior_mask.astype(np.uint8), cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE)
        cv2_module.drawContours(canvas_bgr, contours, -1, PROMPT_COLOR, 1)

    label_anchor = prompt.positive_point or (12, canvas_bgr.shape[0] - 16)
    cv2_module.putText(
        canvas_bgr,
        prompt.prompt_type,
        (label_anchor[0] + 10, max(20, label_anchor[1] - 10)),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        3,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        canvas_bgr,
        prompt.prompt_type,
        (label_anchor[0] + 10, max(20, label_anchor[1] - 10)),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.5,
        PROMPT_COLOR,
        1,
        cv2_module.LINE_AA,
    )
