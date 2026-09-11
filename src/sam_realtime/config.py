from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class ModelEntry:
    name: str
    type: str
    backend: str
    status: str = "planned"
    model_path: Path | None = None
    encoder_path: Path | None = None
    decoder_path: Path | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def paths(self) -> list[Path]:
        primary_paths = [
            path
            for path in [self.model_path, self.encoder_path, self.decoder_path]
            if path is not None
        ]
        metadata_paths = [
            value
            for key, value in self.metadata.items()
            if key.endswith("_path") and isinstance(value, Path)
        ]
        return primary_paths + metadata_paths


@dataclass(frozen=True)
class AppConfig:
    default_model: str
    backend_priority: list[str]
    models: dict[str, ModelEntry]


def load_config(path: Path) -> AppConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    default_model = str(raw.get("default_model", "dummy"))
    backend_priority = [str(item) for item in raw.get("backend_priority", ["cuda", "coreml", "mps", "cpu"])]
    models_raw = raw.get("models", {})
    models: dict[str, ModelEntry] = {}

    for name, entry in models_raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Model entry {name!r} must be a table")
        models[name] = _parse_model_entry(name, entry, path.parent)

    if default_model not in models:
        raise ValueError(f"default_model {default_model!r} is not defined in {path}")

    return AppConfig(default_model=default_model, backend_priority=backend_priority, models=models)


def select_model(config: AppConfig, model_name: str | None) -> ModelEntry:
    selected_name = model_name or config.default_model
    try:
        return config.models[selected_name]
    except KeyError as exc:
        available = ", ".join(sorted(config.models))
        raise ValueError(f"Unknown model {selected_name!r}. Available models: {available}") from exc


def validate_model_files(model: ModelEntry) -> None:
    missing = [path for path in model.paths if not path.exists()]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required model file(s) are missing:\n"
            f"{joined}\n\n"
            "Download or export the model files and update config/models.toml, or run with --model dummy."
        )


def format_model_listing(config: AppConfig) -> str:
    lines = ["Available models:"]
    for name in sorted(config.models):
        model = config.models[name]
        default_marker = " (default)" if name == config.default_model else ""
        lines.append(f"- {name}{default_marker}")
        lines.append(f"  type: {model.type}")
        lines.append(f"  backend: {model.backend}")
        lines.append(f"  status: {model.status}")
        if model.description:
            lines.append(f"  description: {model.description}")
        if model.paths:
            lines.append("  paths:")
            for path in model.paths:
                exists = "present" if path.exists() else "missing"
                lines.append(f"    - {path} [{exists}]")
        if model.metadata.get("source"):
            lines.append(f"  source: {model.metadata['source']}")
        if model.metadata.get("notes"):
            lines.append(f"  notes: {model.metadata['notes']}")
    return "\n".join(lines)


def _parse_model_entry(name: str, entry: dict[str, Any], base_dir: Path) -> ModelEntry:
    known = {"type", "backend", "status", "model_path", "encoder_path", "decoder_path", "description"}
    metadata = {
        key: _metadata_value(key, value, base_dir)
        for key, value in entry.items()
        if key not in known
    }
    model_type = str(entry.get("type", name))
    backend = str(entry.get("backend", "cpu"))

    return ModelEntry(
        name=name,
        type=model_type,
        backend=backend,
        status=str(entry.get("status", "planned")),
        model_path=_optional_path(entry.get("model_path"), base_dir),
        encoder_path=_optional_path(entry.get("encoder_path"), base_dir),
        decoder_path=_optional_path(entry.get("decoder_path"), base_dir),
        description=str(entry.get("description", "")),
        metadata=metadata,
    )


def _metadata_value(key: str, value: Any, base_dir: Path) -> Any:
    if key.endswith("_path"):
        return _optional_path(value, base_dir)
    return value


def _optional_path(value: Any, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
