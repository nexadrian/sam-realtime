# FastSAM-s quantization-aware training experiment

This directory contains the scripts and saved evaluation reports for the
2026-11-04/05 FastSAM-s QAT experiment. Run commands from the repository root.
The checkpoint, captured camera images, training checkpoints, and generated
LiteRT/TFLite models live in the git-ignored `models/` directory. They are not
included here. The scripts are copies of the versions used for the experiment
in `tflite/`.

## What was done

1. Exported the original 512×512 FastSAM-s checkpoint as float and stock INT8
   LiteRT models. Input is a centered square crop resized to 512×512, converted
   from OpenCV BGR to RGB, then divided by 255.
2. Tested PyTorch PT2E QAT with LiteRT Torch. A fake-quantized training step and
   direct LiteRT export worked, but the exported graph had **855 internal float
   operations**. Quantizing that partially quantized graph afterward left 891.
   Neither passes the integer-throughout requirement.
3. Fine-tuned the fused model on `xray` with a frozen float teacher. The full
   model ran for two epochs at batch 4 and learning rate `1e-5`; a separate
   score-head-only run used one epoch at `5e-6`. Training used 250 camera
   calibration images and 100 COCO128-seg images. QAT inserted 222 fake-quant
   modules. The remaining 100 calibration camera images were reserved for
   selection; the original 100-frame holdout and later 40-frame camera set
   were used for final evaluation.
4. Transferred each trained checkpoint back into the fused float architecture,
   exported float LiteRT, and calibrated a **new** static INT8 graph with all
   350 camera calibration images. All three final graphs passed
   `audit_graph.py` and were about 12.25 MB each. This is **QAT-trained weights
   plus post-training quantization**: the PT2E fake-quant scales and zero
   points were not preserved in the final graph.

The training loss covers raw boxes, confidence scores, mask coefficients, and
mask prototypes. Inference applies NMS, prototype decoding, the 0.4 confidence
threshold, and the center-point selection in `fastsam_runtime.py`. Its output
is a stack of binary masks and may contain zero, one, or multiple masks.

## Results

All metrics compare each candidate with the **original float LiteRT model** at
512×512, using the same runtime and thresholds. Count agreement is the share
of frames with an identical number of masks. Median mask IoU pairs masks by
best overlap and assigns zero to unmatched masks, so missing or extra masks
affect it too. The holdout sets were captured separately from the training
split.

| Candidate | Selection count, 100 | Original holdout count, 100 | Original holdout median mask IoU | Later camera count, 40 |
| --- | ---: | ---: | ---: | ---: |
| Stock INT8 | 17% | **48%** | 0.829 | 17.5% |
| Full QAT epoch 1 + PTQ | 31% | 46% | **0.923** | 12.5% |
| Full QAT epoch 2 + PTQ | **34%** | 43% | 0.917 | 10% |
| Score-head QAT + PTQ | 32% | 43% | 0.920 | **20%** |

The QAT candidates improved mask overlap on the original holdout, but did not
consistently improve mask count across scenes. All are well below the 95% count
agreement target. On the original holdout, the QAT-trained **float** checkpoints
had 78%, 71%, and 93% count agreement with the original float model for full
epoch 1, full epoch 2, and score-head respectively. The integer conversion
introduces a further decision gap. These results do not establish a better
deployment model than stock INT8.

Machine-readable per-frame reports are in [`results/`](results/). The same
directory includes `qat_remote_provenance.tgz` with training logs, histories,
and split manifests. Model weights and camera images are omitted because they
are large or private.

## Scripts

| Script | Purpose |
| --- | --- |
| `capture_frames.py` | Capture separated calibration and holdout camera frames (Mac camera index 1 by default). |
| `export_fastsam.py` | Export original float and stock INT8 baselines. |
| `qat_probe.py` | Check PT2E QAT training, direct export, and graph feasibility. |
| `qat_train.py` | Run teacher-distillation QAT on CUDA and save ordinary FastSAM weight checkpoints. |
| `export_qat_weights.py` | Convert QAT-trained weights to float LiteRT, then static INT8 LiteRT. |
| `audit_graph.py` | Reject custom operators and internal floating-point computation. |
| `fastsam_runtime.py` | Run float or integer LiteRT models and return binary masks; requires only LiteRT, OpenCV, and NumPy. |
| `evaluate.py` | Holds the shared mask-comparison metric and original PyTorch comparison. |
| `evaluate_candidates.py` | Compare multiple LiteRT candidates with the original float model. |

`evaluate_candidates.py` imports `compare` from `evaluate.py`, so keep those two
files together. The evaluation/conversion scripts need Ultralytics and PyTorch;
the standalone runtime does not.

## Reproduce

Use the existing `models/.fastsam-converter-venv` on the Mac and the `pt` pyenv
on `xray`, or create environments with compatible versions. This run used
Ultralytics 8.4.161, LiteRT Torch 0.9.4, torchao 0.18.0, AI Edge Quantizer
0.9.0, AI Edge LiteRT 2.2.0, and OpenCV/NumPy. The Mac converter had PyTorch
2.13.0. The `xray` `pt` environment had CUDA PyTorch 2.10.0; torchao warned
that its C++ extensions expected PyTorch 2.11 or newer, although the Python
QAT path ran. For a fresh run, use a separate environment with matching
PyTorch and torchao versions and verify the graph audit again.

### 1. Checkpoint and camera data on the Mac

Download `FastSAM-s.pt` to `models/FastSAM-s.pt` from the
[Ultralytics release](https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-s.pt).
If the existing data is absent, capture separate scenes with these commands:

```sh
models/.fastsam-converter-venv/bin/python fastsam-qat/capture_frames.py --camera 1 --split calibration --count 350
models/.fastsam-converter-venv/bin/python fastsam-qat/capture_frames.py --camera 1 --split holdout --count 100
```

Capture a separate 40-frame `holdout` in
`models/fastsam_camera_fresh_2026-09-25` with `--output-dir` to reproduce the
later-camera evaluation. Use varied objects, distances, lighting, overlap,
and scenes with no object at the center. Fresh captures will give different
scores from the saved reports.

### 2. Baseline exports

```sh
models/.fastsam-converter-venv/bin/python fastsam-qat/export_fastsam.py --size 512 --mode float
models/.fastsam-converter-venv/bin/python fastsam-qat/export_fastsam.py --size 512 --mode int8
models/.fastsam-converter-venv/bin/python fastsam-qat/audit_graph.py models/fastsam_litert/FastSAM-s_512_int8.tflite
```

### 3. Train on `xray`

Prepare `/tmp/fastsam_qat_20261104` on `xray` and copy the scripts, checkpoint,
and calibration images from the Mac:

```sh
ssh xray 'mkdir -p /tmp/fastsam_qat_20261104/data'
scp fastsam-qat/qat_train.py fastsam-qat/qat_probe.py models/FastSAM-s.pt xray:/tmp/fastsam_qat_20261104/
scp -r models/fastsam_camera/calibration xray:/tmp/fastsam_qat_20261104/data/
```

Download and extract the
[COCO128-seg dataset](https://docs.ultralytics.com/datasets/segment/coco128-seg/)
under `data/coco128-seg/`. From `/tmp/fastsam_qat_20261104` run:

```sh
~/.pyenv/versions/pt/bin/python qat_probe.py --checkpoint FastSAM-s.pt --size 512 --device cuda --skip-export
~/.pyenv/versions/pt/bin/python qat_train.py --checkpoint FastSAM-s.pt --camera data/calibration --coco data/coco128-seg/images/train2017 --output output/full --size 512 --batch 4 --epochs 2 --lr 0.00001 --camera-train 250 --coco-train 100
~/.pyenv/versions/pt/bin/python qat_train.py --checkpoint FastSAM-s.pt --camera data/calibration --coco data/coco128-seg/images/train2017 --output output/score_head --size 512 --batch 4 --epochs 1 --lr 0.000005 --trainable score_head --camera-train 250 --coco-train 100
```

The training script sorts file names and uses seed `20261104`. It writes
`data_split.json` and `training_history.json` beside the weights. Compare the
split manifest with the archive in `results/` when reproducing the exact data
order. Copy `output/full/weights_epoch{1,2}.pt` and
`output/score_head/weights_epoch1.pt` back to the Mac, for example under
`models/fastsam_litert/`.

To reproduce the failed direct integer-export gate, run `qat_probe.py` without
`--skip-export`, then run `audit_graph.py` on its output. A nonzero audit exit
code is expected for the direct PT2E export.

### 4. Convert and audit on the Mac

```sh
models/.fastsam-converter-venv/bin/python fastsam-qat/export_qat_weights.py --qat-weights models/fastsam_litert/weights_epoch1.pt --name qat_epoch1 --size 512
models/.fastsam-converter-venv/bin/python fastsam-qat/export_qat_weights.py --qat-weights models/fastsam_litert/weights_epoch2.pt --name qat_epoch2 --size 512
models/.fastsam-converter-venv/bin/python fastsam-qat/export_qat_weights.py --qat-weights models/fastsam_litert/qat_score_head_weights.pt --name qat_score_head --size 512
```

The exporter audits each integer model automatically and fails if internal
float operations remain. It leaves float input and output tensors, which are
allowed by the requirement; model computation is integer.

### 5. Evaluate held-out frames

```sh
models/.fastsam-converter-venv/bin/python fastsam-qat/evaluate_candidates.py \
  --float-model models/fastsam_litert/FastSAM-s_512_float.tflite \
  --candidate stock_int8=models/fastsam_litert/FastSAM-s_512_int8.tflite \
  --candidate qat1_int8=models/fastsam_litert/FastSAM-s_512_qat_epoch1_int8.tflite \
  --candidate qat2_int8=models/fastsam_litert/FastSAM-s_512_qat_epoch2_int8.tflite \
  --candidate score_head_int8=models/fastsam_litert/FastSAM-s_512_qat_score_head_int8.tflite \
  --frames models/fastsam_camera/holdout \
  --output models/fastsam_litert/qat_reproduction_holdout.json
```

For the other sets, change `--frames` to
`models/fastsam_camera_fresh_2026-09-25/holdout` or
`models/fastsam_camera/calibration --start 250 --limit 100`. The latter is the
reserved calibration selection set. The original float model is always the
reference. You can inspect the result visually with
`fastsam_runtime.py --model models/fastsam_litert/FastSAM-s_512_qat_epoch1_int8.tflite --camera 1`.

The main open issue is quantization alignment: direct PT2E export has float
islands, while the integer fallback recalibrates scales after training.
Improving the training objective alone did not solve mask count reliability.
