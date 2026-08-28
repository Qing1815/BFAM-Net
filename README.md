# BFAM-Net

BFAM-Net is a YOLO11-based object detector for impact-crater detection in
lunar south-polar permanently shadowed regions (PSRs). The detector uses four
custom modules:

- **BFRE**: Bidirectional Feature Representation Enhancement. Directional
  state-space scans model long-range horizontal and vertical context.
- **BCFM**: Bidirectional Context Feature Optimization. A channel-context
  branch and a bidirectional spatial state-space branch refine FPN features.
- **RFAM**: Hierarchical Residual Feature Alignment. Residual enhancement,
  MLP aggregation, VGCA-style context aggregation, and depth-aware gating
  align multi-scale features.
- **CSCG**: Contextual Semantic Content Guidance. Learned-offset upsampling
  injects low-resolution context into high-resolution features.

The detector configuration in `configs/bfam-net.yaml` uses three crater scale
classes: `L-crater`, `M-crater`, and `S-crater`.

## Repository Scope

This repository intentionally contains **no dataset, image, annotation,
checkpoint, or experiment result**. Supply a local YOLO-format dataset YAML
when training or validating. The file `configs/lunar-crater.example.yaml`
is only a path and class-name template.

## Installation

Python 3.10 or newer is recommended. Install PyTorch for the target CPU or
CUDA runtime first, then install this repository:

```bash
pip install -e ".[dev]"
```

For ONNX export support:

```bash
pip install -e ".[export]"
```

The model code uses the vendored Ultralytics source tree in this repository.
The project is derived from Ultralytics and remains subject to the AGPL-3.0
license; see `LICENSE`.

## Dataset Format

Prepare a dataset using the standard YOLO detection layout and create a local
dataset YAML based on the example:

```yaml
path: /absolute/path/to/lunar-crater-dataset
train: images/train
val: images/val
test: images/test

names:
  0: L-crater
  1: M-crater
  2: S-crater
```

Each image must have a matching YOLO label file under the corresponding
`labels` directory. The label format is one zero-based class id followed by
normalized `x_center y_center width height` values per line.

## Training

The default training settings are 150 epochs, batch size 16,
640-pixel input, SGD, `lr0=0.01`, momentum `0.937`, weight decay `0.0005`,
and patience 30.

```bash
python scripts/train.py --data /path/to/lunar-crater.yaml --device 0
```

Useful overrides:

```bash
python scripts/train.py \
  --data /path/to/lunar-crater.yaml \
  --epochs 150 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --workers 4
```

Training outputs are written under `runs/`, which is ignored by Git.

## Validation, Prediction, and Export

Validate a trained checkpoint:

```bash
python scripts/val.py \
  --weights runs/bfam-net/train/weights/best.pt \
  --data /path/to/lunar-crater.yaml \
  --device 0
```

Run inference on an image, directory, video, or stream:

```bash
python scripts/predict.py \
  --weights runs/bfam-net/train/weights/best.pt \
  --source /path/to/image-or-directory \
  --conf 0.25 \
  --device 0
```

Export a trained model:

```bash
python scripts/export.py \
  --weights runs/bfam-net/train/weights/best.pt \
  --format onnx \
  --imgsz 640
```

The custom state-space and learned-sampling layers should be tested in the
target export backend before production deployment. Export artifacts are
ignored by Git.

## Model Build Check

The model can be constructed without a dataset or checkpoint:

```bash
python -c "from ultralytics import YOLO; m=YOLO('configs/bfam-net.yaml', verbose=False); m.model.info(imgsz=640)"
```

Run the test suite:

```bash
pytest tests/test_bfam_modules.py
```

The module tests check the output shapes and numerical validity of all four
custom modules, including the grouped sub-pixel offset layout used by CSCG.
The model build check above validates registration and a full detector forward
pass without requiring a dataset or checkpoint.

## License

This repository contains modified Ultralytics source code and custom BFAM-Net
modules. It is distributed under the GNU Affero General Public License,
version 3 or later. See `LICENSE`.
