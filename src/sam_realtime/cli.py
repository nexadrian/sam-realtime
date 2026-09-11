from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .backend import select_backend
from .config import format_model_listing, load_config, select_model, validate_model_files
from .types import RuntimeSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime webcam segmentation experiment.")
    parser.add_argument("--config", type=Path, default=Path("config/models.toml"), help="Path to TOML model registry.")
    parser.add_argument("--list-models", action="store_true", help="List registered models and exit.")
    parser.add_argument("--model", help="Model name from the registry. Defaults to default_model.")
    parser.add_argument("--backend", help="Requested backend/provider. Defaults to registry priority.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--resolution", type=int, default=512, help="Segmentation/display longest side in pixels.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask probability threshold.")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Minimum candidate mask score.")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Overlay opacity from 0.0 to 1.0.")
    parser.add_argument("--output-dir", type=Path, default=Path("captures"), help="Directory for screenshots and masks.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console log level.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    if args.list_models:
        print(format_model_listing(config))
        return 0

    model = select_model(config, args.model)
    backend = select_backend(config.backend_priority, args.backend)

    logger.info("Selected backend: %s (%s) via %s", backend.name, backend.provider, backend.reason)
    if model.paths:
        for path in model.paths:
            logger.info("Selected model: %s path: %s", model.name, path)
    else:
        logger.info("Selected model: %s (no external model files)", model.name)

    validate_model_files(model)

    settings = RuntimeSettings(
        input_long_side=args.resolution,
        mask_threshold=args.mask_threshold,
        score_threshold=args.score_threshold,
        overlay_alpha=args.overlay_alpha,
    )
    from .app import run_app

    return run_app(
        model=model,
        backend=backend,
        camera_index=args.camera,
        settings=settings,
        output_dir=args.output_dir,
    )
