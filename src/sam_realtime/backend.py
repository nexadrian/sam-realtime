from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BackendSelection:
    name: str
    provider: str
    reason: str


def select_backend(priority: list[str], requested: str | None = None) -> BackendSelection:
    candidates = [requested] if requested else priority
    availability = _detect_availability()

    for candidate in candidates:
        if candidate is None:
            continue
        key = candidate.lower()
        if key in availability:
            return availability[key]

    return availability["cpu"]


def onnx_providers_for_backend(backend: BackendSelection | None) -> list[Any]:
    backend_name = backend.name if backend else "cpu"
    if backend_name == "cuda":
        requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif backend_name in {"coreml", "mps"}:
        requested = [_coreml_provider(), "CPUExecutionProvider"]
    else:
        requested = ["CPUExecutionProvider"]

    if importlib.util.find_spec("onnxruntime") is None:
        return requested

    import onnxruntime as ort

    available = set(ort.get_available_providers())
    selected = [
        provider
        for provider in requested
        if _provider_name(provider) in available
    ]
    if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in selected:
        selected.append("CPUExecutionProvider")
    return selected or ["CPUExecutionProvider"]


def _provider_name(provider: Any) -> str:
    if isinstance(provider, tuple):
        return str(provider[0])
    return str(provider)


def _coreml_provider() -> tuple[str, dict[str, str]]:
    cache_dir = Path("models/.ort_coreml_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        "CoreMLExecutionProvider",
        {
            "MLComputeUnits": "ALL",
            "ModelFormat": "MLProgram",
            "RequireStaticInputShapes": "1",
            "ModelCacheDirectory": str(cache_dir),
        },
    )


def _detect_availability() -> dict[str, BackendSelection]:
    available: dict[str, BackendSelection] = {
        "cpu": BackendSelection("cpu", "CPUExecutionProvider", "CPU fallback is always available"),
    }

    if importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime as ort

        providers = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in providers:
            available["cuda"] = BackendSelection("cuda", "CUDAExecutionProvider", "ONNX Runtime CUDA provider is available")
        if "CoreMLExecutionProvider" in providers:
            available["coreml"] = BackendSelection("coreml", "CoreMLExecutionProvider", "ONNX Runtime CoreML provider is available")

    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            available.setdefault("cuda", BackendSelection("cuda", "torch.cuda", "PyTorch CUDA is available"))
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            available["mps"] = BackendSelection("mps", "torch.mps", "PyTorch MPS is available")

    return available
