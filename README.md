# Realtime camera segmentation

This project segments the object at the center of a live camera frame. It
supports several model backends through one OpenCV application. Each frame is
center cropped to a square, resized, and passed to the selected model. The app
displays the highest scoring mask that contains the center point. It does not
accept text prompts or maintain the identity of an object across frames.

The tracked repository contains the application in `src/sam_realtime/`, model
definitions in `config/models.toml`, public asset downloads in
`download_models.sh`, and the selected FastSAM-s QAT LiteRT model in
`models/fastsam_litert/`. Training, export, evaluation, and quantization
details of the FastSAM-s model are in [fastsam-qat/README.md](fastsam-qat/README.md).

## Quick start

Use Python 3.11 or newer. Install the application and LiteRT dependency from
the repository root:

```sh
python -m pip install -e '.[tflite]'
```

The QAT model is included in the repository. Launch it with an OpenCV camera
index available on your system:

```sh
sam-realtime --model fastsam_s_qat_tflite --camera 0 --resolution 512 --score-threshold 0.4
```

`python -m sam_realtime` works in place of `sam-realtime`. Press `q` in the
preview window to quit. To check camera access before downloading any models,
run the synthetic model:

```sh
sam-realtime --model dummy --camera 0
```

Use `sam-realtime --list-models` to list model names and see which expected
files are present. Change `--camera` if the camera you want has a different
OpenCV index, and grant camera access to the application if your operating
system requests it.

### Run without installing this repository

Install only the Python packages needed by the model you want to run. Every
model needs `numpy` and `cv2` (provided by `opencv-python`). Additional
packages may be required depending on the models used. The table below provides
the details.

The following is an example to launch the live segmentation application:

```sh
python -m pip install 'numpy>=1.24' 'opencv-python>=4.8' 'ai-edge-litert>=2.2'
# run the following at repository root
PYTHONPATH=src python -m sam_realtime --model fastsam_s_qat_tflite --camera 0 --resolution 512 --score-threshold 0.4
```

The same command style works for the synthetic model (`dummy`) and the other models
after their files and dependencies are available:

```sh
PYTHONPATH=src python -m sam_realtime --model dummy --camera 0
PYTHONPATH=src python -m sam_realtime --model fastsam_s --camera 0 --resolution 512
PYTHONPATH=src python -m sam_realtime --model efficientsam_onnx --camera 0
PYTHONPATH=src python -m sam_realtime --list-models
```

The commands must run from the repository root, where `src/` and
`config/models.toml` are located. Python 3.11 or newer supplies `tomllib`
from the standard library. For optional models, install these additional
packages; the names in parentheses are the Python modules imported by the app
or the upstream model package:

| Models | Additional packages and imported modules |
| --- | --- |
| `dummy` | None beyond `numpy` and `opencv-python` (`numpy`, `cv2`). |
| All `fastsam_s_*_tflite` models | `ai-edge-litert>=2.2` (`ai_edge_litert`). |
| `efficientsam_onnx`, `mobile_sam_onnx`, `sam_onnx` | `onnxruntime>=1.17` (`onnxruntime`). |
| `fastsam_s` | `ultralytics>=8.3` (`ultralytics`), with its PyTorch dependencies (`torch`, `torchvision`). |
| `sam2_torch` | `torch>=2.3`, `torchvision>=0.18`, `hydra-core` (`torch`, `torchvision`, `hydra`), plus the upstream `sam2` package (`sam2`). |
| `nanosam_trt` | The upstream `nanosam` package (`nanosam`), `Pillow` (`PIL`), and a compatible TensorRT installation (`tensorrt`). Follow NanoSAM's platform-specific setup for its remaining dependencies. |

For example, install the ONNX and FastSAM PyTorch dependencies without
installing this repository:

```sh
python -m pip install 'onnxruntime>=1.17' 'ultralytics>=8.3'
```

## Install models

The QAT model at `models/fastsam_litert/FastSAM-s_512_qat.tflite` is tracked.
Other checkpoints and generated models are local files. The paths below are
resolved relative to `config/models.toml`; place files at those paths or
change the registry.

To download public checkpoints and ONNX files, run the following download script from
the repository root:

```sh
bash download_models.sh
```

The script downloads the original FastSAM-s checkpoint, EfficientSAM-Ti ONNX
files, a MobileSAM ONNX archive, a SAM 2.1 tiny checkpoint and config, and an
original SAM ViT-B checkpoint. It skips files already present. It requires
`curl` and `unzip` and stores the assets under `models/`.

The following table enumerates all models supported:

| Model name | Required files or setup | How to obtain them |
| --- | --- | --- |
| `dummy` | None | Included; produces a synthetic center circle. |
| `fastsam_s_qat_tflite` | `models/fastsam_litert/FastSAM-s_512_qat.tflite` and the `tflite` extra | Model included in this repository. |
| `fastsam_s` | `models/FastSAM-s.pt` and the `fastsam` extra | Run `bash download_models.sh`, or download [FastSAM-s.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt) to that path. |
| `fastsam_s_float_tflite` | `models/fastsam_litert/FastSAM-s_512_float.tflite` and the `tflite` extra | Export the float model with `fastsam-qat/export_fastsam.py` as described below. |
| `fastsam_s_ptq_tflite` | `models/fastsam_litert/FastSAM-s_512_int8.tflite` and the `tflite` extra | Export with camera calibration images using `fastsam-qat/export_fastsam.py`; see the [QAT guide](fastsam-qat/README.md#2-baseline-exports). |
| `efficientsam_onnx` | `models/efficient_sam_encoder.onnx`, `models/efficient_sam_decoder.onnx`, and the `onnx` extra | Run `bash download_models.sh`; files come from [EfficientSAM](https://huggingface.co/yunyangx/EfficientSAM). |
| `mobile_sam_onnx` | `models/mobilesam-onnx-float/` encoder and decoder ONNX files plus their `.data` files, and the `onnx` extra | Run `bash download_models.sh`; it extracts the [Qualcomm MobileSAM](https://huggingface.co/qualcomm/MobileSam) archive. |
| `sam2_torch` | `models/sam2.1_hiera_tiny.pt`, `models/sam2.1_hiera_t.yaml`, PyTorch, and the upstream `sam2` package | Run `bash download_models.sh` for the files; install `sam2` following the [upstream instructions](https://github.com/facebookresearch/sam2). |
| `sam_onnx` | `models/sam_image_encoder.onnx`, `models/sam_mask_decoder.onnx`, and the `onnx` extra | The download script supplies the [SAM ViT-B checkpoint](https://github.com/facebookresearch/segment-anything), but the two ONNX files required by this adapter must still be exported. |
| `nanosam_trt` | `models/nanosam_encoder.engine`, `models/nanosam_decoder.engine`, the upstream `nanosam` package, and TensorRT | Obtain the source models and build engines for the target GPU using [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam). |

## Run the application

The application is a program that accepts several command line arguments.
You can choose a model with `--model`. These examples use the registry's expected
paths and run from the repository root:

```sh
sam-realtime --model fastsam_s_qat_tflite --camera 0 --resolution 512 --score-threshold 0.4
sam-realtime --model fastsam_s --camera 0 --resolution 512
sam-realtime --model efficientsam_onnx --camera 0
sam-realtime --model mobile_sam_onnx --camera 0
sam-realtime --model sam2_torch --camera 0
```

The float and converted INT8 FastSAM models can be selected with
`--model fastsam_s_float_tflite` and `--model fastsam_s_ptq_tflite` after their
files have been generated. `sam_onnx` and `nanosam_trt` need the additional
exported or compiled files listed above before they can run.

The application uses the configured backend priority when no backend is
requested. `--backend` can request `cpu`, `cuda`, `coreml`, or `mps`; if the
requested provider is unavailable, selection falls back to CPU. ONNX models
use ONNX Runtime providers, and PyTorch models use the corresponding PyTorch
device when supported. FastSAM TFLite models use LiteRT's CPU interpreter
regardless of `--backend`.

### Command line options

| Option | Default | Function |
| --- | --- | --- |
| `--config PATH` | `config/models.toml` | Load a different model registry. Relative model paths are resolved from the config file's directory. |
| `--list-models` | Off | Print registered models, required paths, and whether each file exists; then exit. |
| `--model NAME` | `dummy` | Select a model name from the registry. |
| `--backend NAME` | Registry priority | Request `cpu`, `cuda`, `coreml`, or `mps`, subject to model support and installed providers. |
| `--camera ID` | `0` | Select the OpenCV camera index. |
| `--resolution PIXELS` | `512` | Set the side length of the center-cropped square frame used for display and model input where supported. TFLite models have a fixed 512×512 internal input. |
| `--mask-threshold VALUE` | `0.5` | Set the cutoff used to turn mask values into a binary selection. Use a value from 0 to 1. |
| `--score-threshold VALUE` | `0.5` | Reject masks with a lower model score. Use a value from 0 to 1; `0.4` was used for FastSAM TFLite comparisons. |
| `--overlay-alpha VALUE` | `0.45` | Set mask tint opacity in overlay views, from 0 to 1. |
| `--output-dir PATH` | `captures` | Choose where saved screenshots and masks are written. |
| `--log-level LEVEL` | `INFO` | Choose `DEBUG`, `INFO`, `WARNING`, or `ERROR` for console messages. |
| `--help` | — | Show the built-in command help. |

The app finds candidate masks for each frame, keeps those containing the
center point and meeting both thresholds, and displays the highest scoring
one. The on-screen FPS is the full camera loop rate; the overlay also shows
capture, inference, and render times. The `tracking lost` label means there
is no selected mask on the current frame.

### Sliders and keyboard controls

Sliders in the OpenCV window update the following settings while it runs:

| Slider | Range; initial value | Effect |
| --- | --- | --- |
| `mask threshold` | 0–100%; 50% | Sets the binary mask cutoff. |
| `overlay alpha` | 0–100%; 45% | Changes mask tint opacity in overlay and side-by-side views. |
| `input long side` | 64–1536 pixels; 512 | Changes display and preprocessing size. TFLite inference still uses its fixed 512×512 input. |
| `score threshold` | 0–100%; 50% | Rejects lower-scoring masks. |
| `refresh interval` | 1–120; 1 | Displayed control; currently has no effect on inference. |
| `smoothing` | 0–100%; 0% | Displayed control; currently has no effect on masks. |
| `failure grace` | 0–60 frames; 3 | Displayed control; currently has no effect on mask selection. |

The FastSAM PyTorch adapter filters proposals below score 0.4 before the app
applies its `score threshold`, so lowering that slider below 40% cannot bring
those proposals back. Keep `mask threshold` above 0: at 0, the final binary
comparison also accepts zero-valued pixels.

| Key | Action |
| --- | --- |
| `q` | Quit. |
| `v` | Cycle through overlay, isolated object, binary mask, and side-by-side views. |
| `p` | Pause or resume camera capture. The app continues processing the last frame while paused. |
| `r` | Clear the saved mask and box state. |
| `s` | Save the rendered view as a PNG in `--output-dir`; also save the selected binary mask as a PNG when one exists. |

Click the preview window before using keyboard controls if it does not have
focus.

## Quantization of FastSAM-s
This repository also contains the code to quantize the FastSAM-s model into INT8.
The complete calibration, QAT training, export, and requantization steps are in
[fastsam-qat/README.md](fastsam-qat/README.md).

(remove below?)

For `sam2_torch`, install the `torch` extra and the upstream `sam2` package
using its installation instructions. NanoSAM requires its upstream package
and TensorRT in addition to GPU-specific engine files.

The original FastSAM-s checkpoint used for the QAT experiment has SHA-256
`c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844`.
To download only that checkpoint:

```sh
mkdir -p models
curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt -o models/FastSAM-s.pt
```

To generate the optional float FastSAM-s LiteRT baseline after downloading the
checkpoint, use a Python environment with Ultralytics and LiteRT Torch:

```sh
python fastsam-qat/export_fastsam.py --size 512 --mode float
```

The INT8 baseline additionally needs camera calibration frames. The complete
calibration, QAT training, export, and requantization steps are in
[fastsam-qat/README.md](fastsam-qat/README.md).
For `sam2_torch`, install the `torch` extra and the upstream `sam2` package
using its installation instructions. NanoSAM requires its upstream package
and TensorRT in addition to GPU-specific engine files.
