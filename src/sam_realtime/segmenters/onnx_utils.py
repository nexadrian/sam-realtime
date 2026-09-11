from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from ..backend import BackendSelection, onnx_providers_for_backend
from ..config import ModelEntry

MODEL_SIZE = 1024


@dataclass(frozen=True)
class PreparedImage:
    input_tensor: np.ndarray
    scale: float
    resized_size: tuple[int, int]
    original_size: tuple[int, int]


def make_session(path, backend: BackendSelection | None):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime is required for this model. Install the project onnx extra.") from exc

    return ort.InferenceSession(str(path), providers=onnx_providers_for_backend(backend))


def require_paths(model: ModelEntry, *names: str):
    paths = []
    for name in names:
        value = getattr(model, name, None)
        if value is None:
            value = model.metadata.get(name)
        if value is None:
            raise ValueError(f"Model {model.name!r} is missing required registry field {name!r}")
        paths.append(value)
    return paths


def prepare_square_image(
    cv2_module,
    frame_rgb: np.ndarray,
    size: int = MODEL_SIZE,
    *,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> PreparedImage:
    height, width = frame_rgb.shape[:2]
    scale = size / max(height, width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2_module.resize(frame_rgb, (resized_width, resized_height), interpolation=cv2_module.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:resized_height, :resized_width] = resized
    tensor = canvas.astype(np.float32)
    if mean is None or std is None:
        tensor = tensor / 255.0
    else:
        tensor = (tensor - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    tensor = tensor.transpose(2, 0, 1)[None]
    return PreparedImage(
        input_tensor=tensor,
        scale=scale,
        resized_size=(resized_width, resized_height),
        original_size=(width, height),
    )


def point_to_model(point: tuple[int, int], prepared: PreparedImage) -> tuple[float, float]:
    return float(point[0] * prepared.scale), float(point[1] * prepared.scale)


def restore_masks(cv2_module, masks: np.ndarray, prepared: PreparedImage) -> np.ndarray:
    width, height = prepared.original_size
    resized_width, resized_height = prepared.resized_size
    restored = []
    for mask in masks:
        cropped = mask[:resized_height, :resized_width]
        restored.append(cv2_module.resize(cropped.astype(np.float32), (width, height), interpolation=cv2_module.INTER_LINEAR))
    return np.stack(restored, axis=0).astype(np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def timed_call(func):
    start = time.perf_counter()
    result = func()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms
