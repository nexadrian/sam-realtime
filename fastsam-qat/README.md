# Build an INT8 FastSAM-s model

This tutorial builds a 512x512 FastSAM-s LiteRT model with integer internal
computation and measures how well it preserves the original float model's
predictions. The workflow is: establish a PTQ baseline, demonstrate the direct
PT2E export limitations, adapt the model and train QAT, then verify and evaluate
the exported INT8 model.

Run commands from the repository root. Outputs go in `models/fastsam_litert/`;
the final model is `FastSAM-s_512_qat.tflite`. “Integer” refers to the LiteRT
graph: float input/output boundaries are allowed, and decoding and mask
postprocessing run in NumPy outside that graph.

## Environment

The recorded results used Ultralytics 8.4.161, LiteRT Torch 0.9.4, torchao
0.18.0, AI Edge Quantizer 0.9.0, AI Edge LiteRT 2.2.0, and OpenCV/NumPy.
Conversion and evaluation require the PyTorch/Ultralytics environment; QAT
training additionally requires CUDA. Deployment uses LiteRT, OpenCV, and NumPy.
Set the application import path in each shell:

```sh
export PYTHONPATH=src
```

## 1. Establish the PTQ baseline

### Get the checkpoint and images

Download the [FastSAM-s checkpoint](https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt)
and verify its SHA-256:

```sh
mkdir -p models
curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt -o models/FastSAM-s.pt
shasum -a 256 models/FastSAM-s.pt
```

Expected hash: `c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844`.

Capture separate scenes for calibration and evaluation. Change `--camera` to
your camera index; vary objects, distance, lighting, overlap, and empty centers.
The additional 40-frame set provides a second evaluation set. By default, the
captured images are saved to `models/fastsam_camera/calibration` and `models/fastsam_camera/holdout`.

```sh
python fastsam-qat/capture_frames.py --camera 1 --split calibration --count 350
python fastsam-qat/capture_frames.py --camera 1 --split holdout --count 100
python fastsam-qat/capture_frames.py --camera 1 --split holdout --count 40 --output-dir models/fastsam_camera_additional
```

Additionally, download and extract [COCO128-seg](https://docs.ultralytics.com/datasets/segment/coco128-seg/)
so its images are at `data/coco128-seg/images/train2017/`. Training uses images
only in a teacher-student setup; the float teacher supplies targets to the
fake quantization enabled student, so segmentation labels are not read.

| Images (sorted by filename) | Use |
| --- | --- |
| First 250 camera calibration + first 100 COCO images | QAT weight updates |
| Remaining 100 camera calibration images | Candidate selection |
| All 350 camera calibration images | Baseline PTQ calibration |
| Separate 100-frame and additional 40-frame sets | Evaluation only |

The selection set is excluded from QAT weight updates, but participates in
baseline PTQ calibration. Keep evaluation scenes separate from both training
and calibration. Fresh captures will produce different metrics.

### Export, audit, and evaluate

In below, we export the float and INT8 models using LiteRT with the
camera frames as the dataset. Then we audit the INT8 graph to
identify the internal float operations and evaluate it on the
holdout set.

```sh
python fastsam-qat/export_fastsam.py --size 512 --mode float
python fastsam-qat/export_fastsam.py --size 512 --mode int8
python fastsam-qat/audit_graph.py models/fastsam_litert/FastSAM-s_512_int8.tflite --output models/fastsam_litert/ptq.audit.json
python fastsam-qat/evaluate_candidates.py \
  --float-model models/fastsam_litert/FastSAM-s_512_float.tflite \
  --candidate ptq=models/fastsam_litert/FastSAM-s_512_int8.tflite \
  --frames models/fastsam_camera/holdout \
  --output models/fastsam_litert/ptq_holdout.json
```

`audit_graph.py` rejects custom operators and internal operations involving
float tensors, except float input/output conversions. Passing the audit proves
the graph uses integer computation; it does not prove prediction quality.

The recorded PTQ model passed the audit but matched the float model's mask
count on only **48% of the 100 holdout frames**, with median binary-mask IoU
**0.829**. This motivates QAT: valid integer conversion alone does not preserve
segmentation behavior. Metric definitions and the complete comparison are in
section 4.

## 2. Demonstrate the direct PT2E export limitations

PT2E is PyTorch's export-based quantization workflow. `qat_probe.py` fuses
BatchNorm, captures a fixed-size graph with `torch.export()`, and inserts
observers and fake quantization with `prepare_qat_pt2e()`. It takes one SGD
step on a random tensor, then exports through `convert_pt2e()` and
`litert_torch.convert(..., quant_config=QuantConfig(pt2e_quantizer=quantizer))`.
This checks backpropagation and conversion, not segmentation quality.
Each exported probe writes an `.audit.json` report with the float-operation
list and interpreter-allocation result, including any allocation error.

The command below runs the probe to check if the fake quant model can be
exported as INT8 model:

```sh
python fastsam-qat/qat_probe.py --size 512 --output models/fastsam_litert/probe_original.tflite
```

**Expect a nonzero exit in the above command.** The recorded graph has 292 internal float operations,
including SiLU paths lowered as float sigmoid/multiply with quantization
boundaries. It also fails interpreter allocation: its quantized SOFTMAX output
scale is `0.003772`, rather than the required `1/256 = 0.00390625` (zero point
`-128`). This is a separate violation of the
[LiteRT INT8 operator contract](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/quantization_spec.md).

The limitation is in this quantization/conversion path: some patterns are not
lowered to integer operations, and SOFTMAX receives invalid parameters. It
does not mean LiteRT lacks integer sigmoid or arithmetic kernels.

## 3. Adapt the model, train QAT, and export

### Make the graph exportable before training

Two command line arguments are used to make the graph exportable before training:

- `--explicit-silu`: replace all 66 parameter-free `nn.SiLU` modules with
  `x * sigmoid(x)` so the quantizer can recognize the operations.
- `--raw-head`: return box-distribution logits, score logits, mask coefficients,
  and prototypes. Move DFL softmax, anchor/box decoding, score sigmoid, and box
  normalization outside the LiteRT graph.

Run these probes separately; the first is expected to fail, the second to pass:

```sh
python fastsam-qat/qat_probe.py --size 512 --explicit-silu --output models/fastsam_litert/probe_silu.tflite
python fastsam-qat/qat_probe.py --size 512 --explicit-silu --raw-head --output models/fastsam_litert/probe_raw.tflite
```

| 512x512 probe | Internal float operations | Integer audit | Interpreter allocation |
| --- | ---: | --- | --- |
| Original decoded head | 292 | Fail | Fail |
| Explicit SiLU, decoded head | 13 | Fail | Fail |
| Explicit SiLU, raw head | 0 | Pass | Pass |

The 13 remaining operations after the SiLU rewrite are 4 `DEQUANTIZE`, 2 `SUB`,
2 `ADD`, 2 `MUL`, 2 `QUANTIZE`, and 1 `LOGISTIC`, all in decoding or its
boundaries. The invalid quantized SOFTMAX is separate from this float count.
Returning the raw head removes both issues from the exported graph (and shifts
the head computation outside the model).

The SiLU rewrite preserves all 153 checkpoint weight keys; its recorded
maximum float-output difference was below `1.8e-5` on one random 384x384 input.
The NumPy raw-head decoder matched PyTorch's decoded head within `1.8e-7` on
one saved frame, with identical prototypes. These are spot checks of the
transformations, not dataset accuracy measurements.

### Train with a frozen float teacher

Run the following on a CUDA machine with the repository, checkpoint, and image
folders available at the same relative paths. Both runs start independently
from the downloaded checkpoint `FastSAM-s.pt`; the score-head run does not
resume the full-model run.

```sh
# Full model run
python fastsam-qat/qat_train.py \
  --checkpoint models/FastSAM-s.pt \
  --camera models/fastsam_camera/calibration --coco data/coco128-seg/images/train2017 \
  --output models/fastsam_litert/train/full --size 512 --batch 4 \
  --epochs 2 --lr 0.00001 --camera-train 250 --coco-train 100 \
  --explicit-silu --raw-head --save-qat-state

# Score head run
python fastsam-qat/qat_train.py \
  --checkpoint models/FastSAM-s.pt \
  --camera models/fastsam_camera/calibration --coco data/coco128-seg/images/train2017 \
  --output models/fastsam_litert/train/score_head --size 512 --batch 4 \
  --epochs 1 --lr 0.000005 --trainable score_head --camera-train 250 --coco-train 100 \
  --explicit-silu --raw-head --save-qat-state
```

The fake-quant enabled model (student model) simulates INT8 rounding and
clipping while matching the frozen teacher's boxes, scores, mask coefficients,
and prototypes. Raw outputs are decoded in PyTorch for the loss, outside the
exported graph.

Full-model training tests adaptation throughout the network. Score-head
training updates only `model.model.22.cv3.*`, keeping the backbone, box, and
mask weights fixed; fake quantization and observers remain active everywhere.
The short schedules are experimental settings, not an optimal QAT recipe.
Their different learning rates and epoch counts prevent a controlled comparison
of freezing alone.

Each run saves `data_split.json`, `training_history.json`, and checkpoints `qat_state_epoch*.pt` after
each epoch. **Use `qat_state_epoch*.pt` for direct export** as it includes observer
and fake-quantization parameters. The `weights_epoch*.pt` files omits that state. If
training remotely, copy both output directories back to the same paths locally.

### Export and restore required requantization edges

`export_direct_qat.py` rebuilds the fused FastSAM architecture with explicit
SiLU and raw-head outputs, prepares a batch-one PT2E QAT graph, and loads the
trained weights and full fake-quantization state from `qat_state_epoch*.pt`.
It calls `convert_pt2e()` to replace fake quantization with quantize/dequantize
operations using the saved scales and zero points, then converts that graph
to LiteRT with `QuantConfig(pt2e_quantizer=quantizer)`. No new calibration
images are needed. It writes the `.tflite` file and an `.audit.json` report,
rejects internal float or custom operations, and checks interpreter allocation.

`repair_requant.py` reads that LiteRT file and rebuilds the converted PT2E
graph from the same QAT checkpoint to recover the intended convolution-input
scales and zero points. It matches convolutions in execution order, checks
their bias scales against the PT2E input scales, and inserts an INT8-to-INT8
`QUANTIZE` operation wherever a convolution receives the wrong input scale.
It writes a separate repaired `.tflite` file and a `.repair.json` report;
the original export is preserved. This repairs graph connections without
retraining or recalibrating the model.

For this converter version, run both scripts for each candidate:

```sh
for candidate in full_epoch1 full_epoch2 score_head; do
  case "$candidate" in
    full_epoch1) qat_state=models/fastsam_litert/train/full/qat_state_epoch1.pt ;;
    full_epoch2) qat_state=models/fastsam_litert/train/full/qat_state_epoch2.pt ;;
    score_head) qat_state=models/fastsam_litert/train/score_head/qat_state_epoch1.pt ;;
  esac
  model_base="models/fastsam_litert/FastSAM-s_512_qat_${candidate}"
  python fastsam-qat/export_direct_qat.py --qat-state "$qat_state" --size 512 --output "${model_base}_unrepaired.tflite" || break
  python fastsam-qat/repair_requant.py --input "${model_base}_unrepaired.tflite" --output "${model_base}.tflite" --qat-state "$qat_state" --size 512 || break
done
```

#### How the missing requantization was found

The unrepaired direct-QAT model passed the integer audit and LiteRT tensor
allocation, yet produced much lower scores than the same fake-quantized
PyTorch model. We ran the same preprocessed frame through both graphs and
compared their outputs in execution order. The first large difference appeared
at the fourth convolution (LiteRT operator 16), which narrowed the search to
that operator's input, weights, and bias.

For any INT8 `CONV_2D`, we check each channel's quantization parameters:
`bias_scale[channel] ≈ input_scale × weight_scale[channel]`. Operator 16's
actual input scale was 0.10170362, but dividing its bias scale by its weight
scale showed that the bias expected an input scale of 0.06797508. The
convolution therefore applied roughly 0.10170362 / 0.06797508 = 1.496 times
the intended bias. The converted PT2E graph, rebuilt from the saved QAT state,
contained a quantize/dequantize boundary that changed the sliced activation to
scale 0.06797508; the LiteRT graph connected the slice directly to the
convolution. This identified a dropped **INT8-to-INT8 requantization**, not a
missing float operation. Ten convolution inputs had this mismatch in each
recorded direct export; see the `before` array in the `.repair.json` report
generated by the script.

## 4. Measure the final models

All comparisons use the original float LiteRT model as the reference. The
shared runtime center-crops and resizes images to 512x512, converts BGR to RGB,
and divides by 255. Postprocessing uses confidence `0.4`, NMS IoU `0.9`, mask
threshold `0.5`, and retains masks containing the center point. The runtime
automatically decodes raw-head outputs before applying the same postprocessing.

- **Count agreement:** fraction of frames with identical mask counts,
  including frames where both models return zero masks.
- **Median binary-mask IoU:** greedily pair masks by highest overlap, assign
  zero to each unmatched mask, then take the median over all pairs across
  frames. Frames where both models return no masks contribute no IoU pairs.

These measure agreement with the float model, not accuracy against human
annotations. High median overlap can coexist with low count agreement.

The evaluation command:

```sh
python fastsam-qat/evaluate_candidates.py \
  --float-model models/fastsam_litert/FastSAM-s_512_float.tflite \
  --candidate ptq=models/fastsam_litert/FastSAM-s_512_int8.tflite \
  --candidate full_epoch1=models/fastsam_litert/FastSAM-s_512_qat_full_epoch1.tflite \
  --candidate full_epoch2=models/fastsam_litert/FastSAM-s_512_qat_full_epoch2.tflite \
  --candidate score_head=models/fastsam_litert/FastSAM-s_512_qat_score_head.tflite \
  --frames models/fastsam_camera/calibration --start 250 --limit 100 \
  --output models/fastsam_litert/selection.json
```

The command above chooses a candidate using the calibration set. To evaluate the
same candidates on each separate set, rerun the command with these replacements
for the frame selection and output arguments:

| Set | Frame arguments | Output argument |
| --- | --- | --- |
| Holdout, 100 | `--frames models/fastsam_camera/holdout --limit 100` | `--output models/fastsam_litert/holdout.json` |
| Additional holdout, 40 | `--frames models/fastsam_camera_additional/holdout --limit 40` | `--output models/fastsam_litert/additional_holdout.json` |

Remove `--start 250` for both holdouts. Each JSON report includes frame names,
model paths, per-frame comparisons, and aggregate metrics.

Recorded results below are **count agreement / median binary-mask IoU** from
the reference measurements:

| Candidate | Selection, 100 | Holdout, 100 | Additional holdout, 40 |
| --- | --- | --- | --- |
| Stock PTQ | 17% / 0.948 | 48% / 0.829 | 17.5% / 0.922 |
| Repaired QAT, full epoch 1 | 53% / 0.980 | 66% / 0.970 | 52.5% / 0.975 |
| Repaired QAT, full epoch 2 | 50% / 0.982 | 69% / 0.970 | 42.5% / 0.983 |
| Repaired QAT, score head | **60% / 0.981** | 64% / 0.971 | **70% / 0.977** |

Score-head QAT has the highest selection count agreement and is the chosen
candidate. It improves count agreement on both holdouts versus PTQ,
though full-model QAT scores higher on the 100-frame holdout. Updating the score
branch can compensate for substantial quantization error. However, this does not prove
that errors originate only in that branch.

Finally, we can publish the verified score-head candidate to the application's
model path and inspect it with the camera:

```sh
cp models/fastsam_litert/FastSAM-s_512_qat_score_head.tflite models/fastsam_litert/FastSAM-s_512_qat.tflite
python -m sam_realtime --model fastsam_s_qat_tflite --camera 0 --resolution 512 --score-threshold 0.4
```

## Future work

The full-model run measured how much QAT could adapt the network. The
score-head run tested a narrower hypothesis: mask count is sensitive to scores
crossing the `0.4` confidence threshold, so updating the score branch
(`model.model.22.cv3.*`) might help without changing the box and mask weights.
These were two initial choices, not an exhaustive layer search. In particular,
"the last two layers" is ambiguous because FastSAM's box, score, and mask
heads branch apart. The two runs also used different epoch counts and learning
rates, so they do not isolate the effect of freezing weights.

Possible ways to improve mask quality and count agreement include:

- **Compare training scopes under the same conditions.** Train only the final
  score convolution, the entire score branch, all output heads, the final
  backbone stage plus heads, and the full model. Use the same images, seed,
  epochs, learning rate schedule, export path, and thresholds before comparing
  the effects of which weights are trainable.
- **Improve the training objective.** The current loss matches teacher boxes,
  scores, mask coefficients, and prototypes. Test objectives that also reward
  decoded mask overlap and use differentiable surrogates for mask-count
  decisions near the confidence threshold. Segmentation labels could also
  provide a quality target beyond agreement with the float teacher.
- **Expand representative training data and tune quantization.** Include more
  independent scenes, lighting, object sizes, and empty-center frames. Compare
  observer settings and activation clipping or calibration methods where the
  conversion tools support them. Keep the integer-only graph requirement and
  verify intermediate outputs after every export.
- **Use separate data for model selection and final evaluation.** Choose
  training settings and any runtime thresholds on a selection set, then report
  mask-count agreement, binary-mask IoU, and accuracy against annotations on
  untouched holdout scenes. The additional 40-frame holdout is a useful check
  but is too small to establish a stable improvement on its own.
