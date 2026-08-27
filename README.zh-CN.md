# BFAM-Net

BFAM-Net 是一个基于 YOLO11 的月球南极永久阴影区（PSR）陨石坑检测器，包含四个
自定义模块：

- **BFRE**：双向特征表示增强模块，沿水平和垂直方向进行双向状态空间扫描，
  建模长程空间上下文。
- **BCFM**：双向上下文特征优化模块，通过通道上下文分支和双向空间状态空间分支，
  细化 FPN 特征。
- **RFAM**：层次残差特征对齐模块，结合残差增强、MLP 聚合、VGCA 风格上下文聚合和
  深度感知门控，改善多尺度特征对齐。
- **CSCG**：上下文语义内容引导模块，通过学习偏移的动态上采样，将低分辨率语义
  信息注入高分辨率特征。

`configs/bfam-net.yaml` 使用三类陨石坑：`L-crater`、`M-crater` 和 `S-crater`。

## 仓库范围

本仓库**不包含数据集、图像、标注、模型权重或实验结果**。训练和验证时请通过
命令行传入本地 YOLO 数据集 YAML。`configs/lunar-crater.example.yaml` 只是路径和
类别名称模板。

## 安装

建议使用 Python 3.10 或更高版本。请先根据目标 CPU 或 CUDA 环境安装 PyTorch，
然后在仓库根目录执行：

```bash
pip install -e ".[dev]"
```

需要 ONNX 导出支持时执行：

```bash
pip install -e ".[export]"
```

模型代码使用仓库内的 Ultralytics 源码。该代码由 Ultralytics 派生并修改，仍受
AGPL-3.0 许可证约束，详见 `LICENSE` 和 `NOTICE.md`。

## 数据集格式

准备标准 YOLO 检测数据目录，并参考示例创建本地数据集 YAML：

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

每张图像都应在对应的 `labels` 目录下有同名标注文件。每行格式为：从 0 开始的
类别编号，以及归一化的 `x_center y_center width height`。

## 训练

训练脚本默认设置为：150 个 epoch、batch size 为 16、输入尺寸
640、SGD、`lr0=0.01`、momentum `0.937`、weight decay `0.0005`、patience 30。

```bash
python scripts/train.py --data /path/to/lunar-crater.yaml --device 0
```

输出保存在 `runs/` 下，该目录已加入 Git 忽略规则。

## 验证、推理和导出

```bash
python scripts/val.py \
  --weights runs/bfam-net/train/weights/best.pt \
  --data /path/to/lunar-crater.yaml \
  --device 0
```

```bash
python scripts/predict.py \
  --weights runs/bfam-net/train/weights/best.pt \
  --source /path/to/image-or-directory \
  --conf 0.25 \
  --device 0
```

```bash
python scripts/export.py \
  --weights runs/bfam-net/train/weights/best.pt \
  --format onnx \
  --imgsz 640
```

自定义状态空间层和学习型采样层在不同导出后端上的支持情况可能不同，部署前应在
目标后端完成测试。导出文件已加入 Git 忽略规则。

## 构建和测试

不需要数据集或权重即可构建模型：

```bash
python -c "from ultralytics import YOLO; m=YOLO('configs/bfam-net.yaml', verbose=False); m.model.info(imgsz=640)"
```

运行测试：

```bash
pytest tests/test_bfam_modules.py
```

模块测试会检查四个自定义模块的输出形状、数值有效性，以及 CSCG 分组子像素偏移布局。上面的模型构建命令会在不需要
数据集或权重的情况下检查模块注册，并执行一次完整检测器前向传播。

## 许可证

本仓库包含修改后的 Ultralytics 源代码和 BFAM-Net 自定义模块，按 GNU Affero
通用公共许可证第三版或更高版本发布。详见 `LICENSE` 和 `NOTICE.md`。
