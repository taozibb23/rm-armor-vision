# 01 训练：YOLO26-pose 装甲板关键点模型

本目录包含数据集配置、训练命令、导出脚本，以及本次训练（100 epoch）的全部产物与指标。

---

## 1. 任务定义

用 ultralytics 的 **pose** 任务做装甲板检测 + 4 角点定位：

- **输入**：一张 RGB 图（640×640）
- **输出**：每个装甲板的
  - 检测框（类别 + 位置）
  - **4 个角点**：`left_top` / `left_bottom` / `right_bottom` / `right_top`

**为什么要角点**：PnP 测距与位姿估计需要 4 个角点的像素坐标，检测框只能给出位置，角点才能给出姿态。

---

## 2. 数据集

| 项 | 内容 |
|---|---|
| 名称 | XJTLU 2023 Keypoints（公开数据集） |
| 规模 | 共 6732 张：**train 4224 / val 2508** |
| 类别 | **14 类**：`B1~B5`、`BO`、`BS`、`R1~R5`、`RO`、`RS` |
| 关键点 | **4 个**，`kpt_shape=[4, 2]` |
| 标注格式 | YOLO pose（`class cx cy w h  kx1 ky1 v1 ...`） |

**类别命名**：`R` = 红方，`B` = 蓝方，`O` = 前哨站，`S` = 基地；`1~5` 为装甲板编号。

**关键点顺序**（定义在 `dataset.yaml` 中）：

```
0: left_top      1: left_bottom      2: right_bottom      3: right_top
```

该顺序在训练标注、推理解码、PnP 求解三处必须保持一致，错一位会导致位姿结果完全错误。

数据集配置见 [`dataset.yaml`](dataset.yaml)（`path` 需指向本地数据集目录）。

---

## 3. 模型结构

- 使用 ultralytics 内置的 **`cfg/models/26/yolo26-pose.yaml`**（**n** 尺度）
- 训练时 ultralytics 会依据数据集**自动覆盖**默认配置，日志中可见：

```
Overriding model.yaml kpt_shape=[17, 3] with kpt_shape=[4, 2]
Overriding model.yaml nc=80 with nc=14
```

| 配置项 | 默认（COCO 人体姿态） | 本项目 |
|---|---|---|
| 类别数 `nc` | 80 | **14** |
| 关键点 `kpt_shape` | `[17, 3]`（17 点 + 可见性） | **`[4, 2]`**（4 点，无可见性） |

**最终模型规模**：`129 layers, 2,449,494 parameters, 5.6 GFLOPs`（约 244 万参数，`.pt` 文件 5.9MB）

---

## 4. 训练命令

完整命令见 [`train.sh`](train.sh)，核心参数：

```bash
yolo pose train \
  model=yolo26-pose.yaml \
  data=dataset.yaml \
  epochs=100 imgsz=640 batch=-1 device=0 workers=8 \
  name=armor_yolo26
```

**环境**：Ubuntu 22.04 · Python 3.10.12 · ultralytics **8.4.142** · torch 2.14.0+cu130 · RTX 4060 Laptop（8G）

**耗时**：100 epoch 约 **4070 秒（68 分钟）**

---

## 5. 训练结果

### 5.1 总体指标（val：2508 张 / 3225 个实例）

| 任务 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 检测框 | 0.850 | 0.760 | **0.842** | 0.559 |
| 角点 | 0.744 | 0.662 | **0.675** | 0.610 |

### 5.2 逐类结果

| 类别 | 实例数 | Box mAP50 | Pose mAP50 | Pose mAP50-95 |
|---|---|---|---|---|
| B1 | 313 | 0.870 | 0.618 | 0.588 |
| B2 | 362 | 0.916 | 0.888 | 0.853 |
| B3 | 246 | 0.792 | 0.589 | 0.518 |
| B4 | 454 | 0.881 | 0.719 | 0.642 |
| B5 | 248 | 0.779 | 0.690 | 0.635 |
| BO | 70 | 0.921 | 0.467 | 0.458 |
| R1 | 225 | 0.730 | 0.710 | 0.661 |
| R2 | 354 | 0.873 | 0.853 | 0.705 |
| R3 | 196 | 0.843 | 0.837 | 0.831 |
| R4 | 335 | 0.801 | 0.724 | 0.612 |
| R5 | 283 | 0.897 | **0.329** | **0.215** |
| RO | 139 | 0.806 | 0.678 | 0.600 |

**两个明显的异常**：

- **R5**：检测效果很好（Box mAP50 = 0.897），但角点精度显著偏低（Pose mAP50 = 0.329）。
  检测与关键点共用同一套特征，检测正常而关键点差，通常指向**该类样本的关键点标注质量或数量问题**，而非模型容量不足。这是后续最值得排查的方向。
- **BO**：仅 70 个实例，样本量偏少，角点 mAP50 = 0.467。

### 5.3 训练产物

| 文件 | 内容 |
|---|---|
| [`results.png`](results/results.png) | 训练曲线（loss 下降 / mAP 上升） |
| [`results.csv`](results/results.csv) | 100 个 epoch 的逐轮指标原始数据 |
| [`args.yaml`](results/args.yaml) | 本次训练的全部超参数 |
| [`BoxPR_curve.png`](results/BoxPR_curve.png) / [`PosePR_curve.png`](results/PosePR_curve.png) | 检测框 / 角点的 PR 曲线 |
| [`confusion_matrix_normalized.png`](results/confusion_matrix_normalized.png) | 归一化混淆矩阵 |
| [`labels.jpg`](results/labels.jpg) | 数据集分布（框大小、位置、长宽比） |
| [`train_batch0.jpg`](results/train_batch0.jpg) | 训练数据增强效果（mosaic 拼图） |
| [`val_batch0_labels.jpg`](results/val_batch0_labels.jpg) | 验证集标注真值 |
| [`val_batch0_pred.jpg`](results/val_batch0_pred.jpg) | 验证集模型预测 |

---

## 6. 训练方式：本次为从随机初始化开始

`args.yaml` 中的 `pretrained: true` 是 ultralytics 的**默认参数**，本次训练**未实际加载任何预训练权重**。依据有两条：

### 依据一：第 1 个 epoch 的 mAP 接近 0

| epoch | Box mAP50 | Pose mAP50 |
|---|---|---|
| 1 | 0.00084 | 0.00078 |
| 2 | 0.03615 | 0.02123 |
| 3 | 0.06171 | 0.04134 |
| … | … | … |
| 100 | 0.842 | 0.675 |

若加载了预训练权重，第 1 个 epoch 的 mAP 不会处于 0.0008 这一量级。原始数据见 [`results.csv`](results/results.csv) 第 1 行。

### 依据二：日志中没有权重加载记录

ultralytics 在成功加载预训练权重时会输出 `Transferred xxx/xxx items from pretrained weights`。
本次训练日志中该记录出现 **0 次**，且工程目录下不存在任何 `yolo26-pose.pt`。

### 复现方式

按 [`train.sh`](train.sh) 执行即可——使用结构文件（`.yaml`）而非权重文件（`.pt`）作为 `model` 参数。

---

## 7. 导出 ONNX

见 [`export_onnx.sh`](export_onnx.sh)：

```bash
yolo export model=weights/best.pt format=onnx imgsz=640 opset=12 simplify=True
```

导出结果：`best.onnx` 约 **10.2MB**，输入 `(1,3,640,640)`，输出 `(1,26,8400)`。

**输出 26 行的含义**（推理侧解码依据，详见 [`../02-onnx-inference/`](../02-onnx-inference/)）：

```
行 0~3   框：cx, cy, w, h
行 4~17  14 个类别分数        ← 类号 = 行号 − 4
行 18~25 4 个角点的 (x, y)    ← 每 2 行一个点
8400     候选数量 = 80×80 + 40×40 + 20×20 = 6400 + 1600 + 400
```

---

## 8. 训练阶段遇到的问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 显存不足 | `batch=-1` 自动选择的批大小超出 8G 显存 | 手动指定 `batch=16` 或更小 |
| PnP 结果整体错误 | 训练标注的角点顺序与推理代码中的顺序不一致 | 以 `dataset.yaml` 的 `kpt_names` 为唯一标准，三处保持一致 |
| 类别数 / 关键点数不符 | 结构文件默认为 COCO 的 80 类 / 17 点 | 由 `dataset.yaml` 自动覆盖；需确认日志中出现两行 `Overriding` |

---

## 9. 目录内容

```
01-yolo-training/
├── README.md          本文件
├── dataset.yaml       数据集配置
├── train.sh           训练命令
├── export_onnx.sh     导出 ONNX
└── results/           本次训练产物
```

**权重文件未入库**（`best.pt` / `best.onnx`，体积原因）。复现方式见第 6 节与第 7 节。
