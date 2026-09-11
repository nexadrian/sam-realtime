from __future__ import annotations

from collections import deque

import numpy as np

from .types import PromptEvidence, SelectedMask


def choose_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    prompt: PromptEvidence,
    *,
    mask_threshold: float,
    score_threshold: float,
) -> SelectedMask | None:
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape (N, H, W), got {masks.shape}")
    if len(scores) != masks.shape[0]:
        raise ValueError("scores length must match number of masks")

    prompt_point = prompt.positive_point
    candidates: list[SelectedMask] = []

    for mask, score_value in zip(masks, scores, strict=True):
        score = float(score_value)
        if score < score_threshold:
            continue
        binary = mask >= mask_threshold
        cleaned = largest_component_containing_point(binary, prompt_point)
        if cleaned is None:
            continue
        candidates.append(SelectedMask(mask=cleaned, score=score, area=int(cleaned.sum())))

    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.score)


def largest_component_containing_point(mask: np.ndarray, point: tuple[int, int] | None) -> np.ndarray | None:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")

    binary = mask.astype(bool)
    if not binary.any():
        return None

    if point is not None:
        x, y = point
        if 0 <= y < binary.shape[0] and 0 <= x < binary.shape[1] and binary[y, x]:
            return _flood_component(binary, x, y)
        return None

    return _largest_component(binary)


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _largest_component(mask: np.ndarray) -> np.ndarray:
    visited = np.zeros(mask.shape, dtype=bool)
    best: np.ndarray | None = None
    best_area = 0

    for y, x in zip(*np.where(mask), strict=True):
        if visited[y, x]:
            continue
        component = _flood_component(mask, int(x), int(y), visited)
        area = int(component.sum())
        if area > best_area:
            best = component
            best_area = area

    if best is None:
        return np.zeros(mask.shape, dtype=bool)
    return best


def _flood_component(
    mask: np.ndarray,
    start_x: int,
    start_y: int,
    visited: np.ndarray | None = None,
) -> np.ndarray:
    if visited is None:
        visited = np.zeros(mask.shape, dtype=bool)
    component = np.zeros(mask.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
    visited[start_y, start_x] = True

    while queue:
        x, y = queue.popleft()
        if not mask[y, x]:
            continue
        component[y, x] = True

        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or ny >= mask.shape[0] or nx >= mask.shape[1]:
                continue
            if visited[ny, nx]:
                continue
            visited[ny, nx] = True
            if mask[ny, nx]:
                queue.append((nx, ny))

    return component
