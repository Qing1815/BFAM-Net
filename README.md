# BFAM-Net

BFAM-Net is a YOLO11-based object detector for impact-crater detection in
lunar south-polar permanently shadowed regions (PSRs). The repository is a
cleaned, source-tree implementation of the final module names used in the
manuscript:

- **BFRE**: Bidirectional Feature Representation Enhancement. Directional
  state-space scans model long-range horizontal and vertical context.
- **BCFM**: Bidirectional Context Feature Optimization. A channel-context
  branch and a bidirectional spatial state-space branch refine FPN features.
- **RFAM**: Hierarchical Residual Feature Alignment. Residual enhancement,
  MLP aggregation, VGCA-style context aggregation, and depth-aware gating
  align multi-scale features.
- **CSCG**: Contextual Semantic Content Guidance. Learned-offset upsampling
  injects low-resolution context into high-resolution features.

The detector configuration in `configs/bfam-net.yaml` uses the three crater
scale classes reported in the manuscript: `L-crater`, `M-crater`, and
`S-crater`.

## Repository Scope

This repository intentionally contains **no dataset, image, annotation,
checkpoint, or experiment result**. Supply a local YOLO-format dataset YAML
when training or validating. The file `configs/lunar-crater.example.yaml`
is only a path and class-name template.

The original experiment directory did not contain registered `BFRE`, `BCFM`,
or `CSCG` classes under those final names. It contained related experimental
implementations named `BFEM`, `DFRM`, and `CCGM`. This repository reconstructs
the final BFAM-Net interfaces from the manuscript and consolidates the
corresponding reference implementations into `ultralytics/nn/modules/bfam.py`.
It is therefore a cleaned reproducibility implementation, not a
binary-identical export of the original experiment directory.

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
license; see `LICENSE` and `NOTICE.md`.

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

The manuscript settings are used by default: 150 epochs, batch size 8,
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
  --batch 8 \
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
pytest tests/test_bfam_modules.py tests/smoke_test.py
```

The smoke test checks that all four custom modules are registered, the model
has `8/16/32` output strides, and a random tensor can pass through the full
detector.

## Manuscript Results

The manuscript reports BFAM-Net results of 4.75M parameters, 11.40 GFLOPs,
78.73 FPS, and 94.2% mAP on its Impact_crater_datas evaluation. It reports
BFAM-Tiny at 93.1% mAP and 6.14 MB. These numbers are **manuscript-reported
results** and are not claimed as a re-run result of this repository. Exact
matching requires the original data split, preprocessing, augmentation,
training environment, pruning/distillation recipe, and trained weights,
none of which are included here.

## Reference Basis

The custom modules were organized from the manuscript and the three local
reference papers supplied with the project:

1. Hu, Z., Zhai, B., Zhao, Z., et al. “State-Space-Model-Guided Deep Feature
   Perception Network for Insulator Defect Detection in High-Resolution Aerial
   Images.” *IEEE Transactions on Geoscience and Remote Sensing*, 63 (2025).
   DOI: `10.1109/TGRS.2025.3584663`. This is the primary reference for BFRE
   and BCFM concepts, corresponding to BFEM and DFRM in the paper.
2. Wu, S., Lu, X., Guo, C., and Guo, H. “MV-YOLO: An Efficient Small Object
   Detection Framework Based on Mamba.” *IEEE Transactions on Geoscience and
   Remote Sensing*, 63 (2025). DOI: `10.1109/TGRS.2025.3584955`. This is the
   primary reference for contextual semantic guidance and dynamic upsampling,
   corresponding to CCGM/DySample in the paper.
3. Wang, J. and Yan, C. “CEVG-RTNet: A Real-Time Architecture for Robust
   Forest Fire Smoke Detection in Complex Environments.” *Neural Networks*,
   194 (2026), 108187. DOI: `10.1016/j.neunet.2025.108187`. This is the
   primary reference for the hierarchical residual feature alignment design,
   especially ERFB, VGCA-style aggregation, MLP enhancement, and gating.

The local PDF copies are not redistributed in this code repository.

## License

This repository contains modified Ultralytics source code and custom BFAM-Net
modules. It is distributed under the GNU Affero General Public License,
version 3 or later. See `LICENSE` and `NOTICE.md` for the derivation notice.

## Citation

The manuscript citation details were not supplied as a finalized bibliographic
record. Add the accepted paper's author list, title, venue, year, and DOI here
before publishing a release citation. Do not infer missing citation metadata.
