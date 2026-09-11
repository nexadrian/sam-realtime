#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
MODELS_DIR="$ROOT_DIR/models"

if [[ ! -f "$ROOT_DIR/pyproject.toml" || ! -f "$ROOT_DIR/config/models.toml" ]]; then
  echo "error: run this script from the root directory of the sam-realtime repo" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required" >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "error: unzip is required" >&2
  exit 1
fi

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

mkdir -p "$MODELS_DIR"

download() {
  local url="$1"
  local dest="$2"
  local label="$3"

  mkdir -p "$(dirname "$dest")"

  if [[ -s "$dest" && "$FORCE" != "1" ]]; then
    echo "skip: $label already exists at ${dest#$ROOT_DIR/}"
    return
  fi

  if [[ "$FORCE" == "1" ]]; then
    rm -f "$dest"
  fi

  echo "download: $label"
  echo "  from: $url"
  echo "  to:   ${dest#$ROOT_DIR/}"
  curl \
    --location \
    --fail \
    --retry 3 \
    --retry-delay 2 \
    --continue-at - \
    --output "$dest" \
    "$url"
}

download \
  "https://huggingface.co/yunyangx/EfficientSAM/resolve/main/efficientsam_ti_encoder.onnx" \
  "$MODELS_DIR/efficient_sam_encoder.onnx" \
  "EfficientSAM-Ti ONNX encoder"

download \
  "https://huggingface.co/yunyangx/EfficientSAM/resolve/main/efficientsam_ti_decoder.onnx" \
  "$MODELS_DIR/efficient_sam_decoder.onnx" \
  "EfficientSAM-Ti ONNX decoder"

download \
  "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/mobilesam/releases/v0.62.1/mobilesam-onnx-float.zip" \
  "$MODELS_DIR/mobilesam-onnx-float.zip" \
  "MobileSAM Qualcomm ONNX export zip"

if [[ "$FORCE" == "1" ]]; then
  unzip -o "$MODELS_DIR/mobilesam-onnx-float.zip" -d "$MODELS_DIR"
else
  unzip -n "$MODELS_DIR/mobilesam-onnx-float.zip" -d "$MODELS_DIR"
fi

download \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" \
  "$MODELS_DIR/sam2.1_hiera_tiny.pt" \
  "SAM2.1 tiny checkpoint"

download \
  "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml" \
  "$MODELS_DIR/sam2.1_hiera_t.yaml" \
  "SAM2.1 tiny config"

download \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
  "$MODELS_DIR/sam_vit_b_01ec64.pth" \
  "Original SAM ViT-B checkpoint"

download \
  "https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-s.pt" \
  "$MODELS_DIR/FastSAM-s.pt" \
  "FastSAM-s checkpoint"
