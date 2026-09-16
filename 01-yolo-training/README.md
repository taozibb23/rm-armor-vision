# 01 训练：从零训练 YOLO26-pose 装甲板关键点模型

> 这一节的目的是证明：**模型是自己训练出来的**，不是下载别人的权重跑 demo。

---

## 1. 任务定义

装甲板检测 + 4 角点定位（关键点检测），用 ultralytics 的 **pose** 任务：

- **输入**：一张 RGB 图（640×640）
- **输出**：每个装甲板的
  - 检测框（类别 + 位置）
  - **4 个角点**：`left_top` / `left_bottom` / `right_bottom` / `right_top`

**为什么要角点**：后续 PnP 测距/测位姿需要 4 个角点的像素坐标（见 `04` / `05` 节）。

---

## 2. 数据集

| 项 | 内容 |
|---|---|
| 名称 | XJTLU 2023 Keypoints（公开数据集） |
| 规模 | 共 6732 张：**train 4224 / val 2508** |
| 类别 | **14 类**：`B1~B5`、`BO`、`BS`、`R1~R5`、`RO`、`RS` |
| 关键点 | **4 个**，`kpt_shape=[4, 2]` |
| 标注格式 | YOLO pose（`class cx cy w h  kx1 ky1 v1 ...`） |

**类别命名**（R = 红方，B = 蓝方，O = 前哨站，S = 基地）：
`R1~R5` / `B1~B5` = 五种装甲板编号，`RO`/`BO` = 前哨站，`RS`/`BS` = 基地。

**关键点顺序**（`dataset.yaml` 里定义）：

```
0: left_top      1: left_bottom      2: right_bottom      3: right_top
```

**顺序必须固定**——后面 PnP 时会把它和真实世界坐标一一对应，顺序错一位结果全错。

配置文件见 [`dataset.yaml`](dataset.yaml)（路径需按自己的环境改）。

---

## 3. 模型结构

- 用 ultralytics 内置的 **`cfg/models/26/yolo26-pose.yaml`**（**n** 尺度）
- 训练时 ultralytics 会**按数据集自动覆盖**默认值，日志里能看到：

```
Overriding model.yaml kpt_shape=[17, 3] with kpt_shape=[4, 2]
Overriding model.yaml nc=80 with nc=14
```

| 覆盖项 | 默认（COCO 人体姿态） | 本项目 |
|---|---|---|
| 类别数 `nc` | 80 | **14** |
| 关键点 `kpt_shape` | `[17, 3]`（17 点 + 可见性） | **`[4, 2]`**（4 点，无可见性） |

**最终模型**：`YOLO26-pose summary (fused): 129 layers, 2,449,494 parameters, 5.6 GFLOPs`
（约 **244 万**参数，5.9MB 的 `.pt`）

---

## 4. 训练命令

见 [`train.sh`](train.sh)。核心是：

```bash
yolo pose train \
  model=yolo26-pose.yaml \
  data=dataset.yaml \
  epochs=100 imgsz=640 batch=-1 device=0 workers=8 \
  name=armor_yolo26
```

**实际环境**：Ubuntu 22.04 + Python 3.10.12 + ultralytics **8.4.142** + torch 2.14.0+cu130 + RTX 4060 Laptop（8G）

**全程约 4070 秒（≈68 分钟）**，100 epoch。

---

## 5. 训练结果

### 5.1 总体指标（val 2508 张 / 3225 个实例）

| 任务 | Precision | Recall | **mAP50** | **mAP50-95** |
|---|---|---|---|---|
| **检测框** | 0.850 | 0.760 | **0.842** | **0.559** |
| **角点** | 0.744 | 0.662 | **0.675** | **0.610** |

### 5.2 逐类结果（这是最能看出问题的一张表）

| 类别 | 实例数 | Box mAP50 | Pose mAP50 | Pose mAP50-95 |
|---|---|---|---|---|
| B1 | 313 | 0.870 | 0.618 | 0.588 |
| B2 | 362 | 0.916 | **0.888** | **0.853** |
| B3 | 246 | 0.792 | 0.589 | 0.518 |
| B4 | 454 | 0.881 | 0.719 | 0.642 |
| B5 | 248 | 0.779 | 0.690 | 0.635 |
| BO | 70 | **0.921** | 0.467 | 0.458 |
| R1 | 225 | 0.730 | 0.710 | 0.661 |
| R2 | 354 | 0.873 | 0.853 | 0.705 |
| R3 | 196 | 0.843 | 0.837 | **0.831** |
| R4 | 335 | 0.801 | 0.724 | 0.612 |
| R5 | 283 | 0.897 | **0.329** ⚠️ | **0.215** ⚠️ |
| RO | 139 | 0.806 | 0.678 | 0.600 |

**明显的问题**：
- **R5 的检测很好（mAP50 0.897），但角点很差（0.329）** → 说明 R5 这类样本的关键点标注质量或数量有问题。**这也是下一步最该优化的方向**（不是调超参，而是去查 R5 的标注）。
- `BO`（前哨站，只有 70 个实例）样本太少，角点 0.467。

### 5.3 训练曲线与图

| 文件 | 内容 |
|---|---|
| [`results.png`](results/results.png) | 训练曲线（loss 下降 + mAP 上升） |
| [`results.csv`](results/results.csv) | 100 个 epoch 的逐轮数字（可自己画图） |
| [`args.yaml`](results/args.yaml) | 本次训练的**全部超参数**（可复现） |
| [`BoxPR_curve.png`](results/BoxPR_curve.png) | 检测框 PR 曲线 |
| [`PosePR_curve.png`](results/PosePR_curve.png) | 角点 PR 曲线 |
| [`confusion_matrix_normalized.png`](results/confusion_matrix_normalized.png) | 归一化混淆矩阵（看哪些类互相混） |
| [`labels.jpg`](results/labels.jpg) | 数据集分布（框大小/位置/长宽比） |
| [`val_batch0_labels.jpg`](results/val_batch0_labels.jpg) | 验证集**真值** |
| [`val_batch0_pred.jpg`](results/val_batch0_pred.jpg) | 验证集**预测**（和真值对比看） |
| [`train_batch0.jpg`](results/train_batch0.jpg) | 训练时的增强效果（mosaic 拼图） |

---

## 6. ⭐ "从零训练"的证据（如果被问到，用这两条回答）

**这是本项目最容易被质疑的一点，所以留了证据：**

### 证据一：第 1 个 epoch 的 mAP 几乎为 0

```
epoch 1:  Box mAP50 = 0.00084    Pose mAP50 = 0.00078
epoch 2:  Box mAP50 = 0.03615    Pose mAP50 = 0.02123
epoch 3:  Box mAP50 = 0.06171    Pose mAP50 = 0.04134
...
epoch 100: Box mAP50 = 0.842     Pose mAP50 = 0.675
```

**如果加载了预训练权重，第 1 个 epoch 的 mAP 不可能是 0.0008** —— 那说明权重是从**随机初始化**开始的。
（数据见 [`results.csv`](results/results.csv)，可自行核对第一行。）

### 证据二：日志里没有任何"加载预训练权重"的记录

ultralytics 在成功加载预训练权重时会打印 `Transferred xxx/xxx items from pretrained weights`。
本次训练日志中**这条记录出现 0 次**，且磁盘上不存在任何 `yolo26-pose.pt`。

### ⚠️ 一个要主动说清的细节

`args.yaml` 里有 `pretrained: true` —— 这是 **ultralytics 的默认值**。
**但实际没有加载到任何权重**（证据如上）。如果面试官看到这个参数问起，就这样答：

> "`pretrained: true` 是 ultralytics 的默认参数。因为我用的是 `yolo26-pose.yaml` 结构文件、本地也没有对应的 `.pt`，实际是从随机初始化开始训练的——`results.csv` 第 1 行的 mAP 只有 0.0008，日志里也没有 `Transferred from pretrained weights` 的记录。"

**主动说清比被问出来好。**

---

## 7. 导出 ONNX

```bash
yolo export model=runs/pose/armor_yolo26/weights/best.pt \
           format=onnx imgsz=640 opset=12 simplify=True
```

导出后 `best.onnx` 约 **10.2MB**，输入 `(1,3,640,640)`，输出 `(1,26,8400)`。

**输出 26 行的含义**（推理侧解码用，详见 `02-onnx-inference/`）：

```
行 0~3   : 框（cx, cy, w, h）
行 4~17  : 14 个类别的分数        ← 类号 = 行号 − 4
行 18~25 : 4 个角点的 (x, y)      ← 每 2 行一个点
8400     : 候选框数量（80×80 + 40×40 + 20×20 = 6400+1600+400）
```

---

## 8. 踩坑记录

| 问题 | 原因 | 解决 |
|---|---|---|
| 训练起不来，报显存不足 | `batch=-1` 自动选批太大 | 手动指定 `batch=16` |
| 关键点顺序对不上，PnP 结果全错 | 训练标注的角点顺序和推理时代码里写的顺序不一致 | **以 `dataset.yaml` 的 `kpt_names` 为唯一标准**，代码里照抄 |
| 类别数/关键点数不对 | 结构文件默认是 COCO 的 80 类 / 17 点 | 靠 `dataset.yaml` 自动覆盖（见第 3 节），**但要检查日志里有没有这两行 Overriding** |

---

## 9. 目录内容

```
01-yolo-training/
├── README.md          本文件
├── dataset.yaml       数据集配置（路径需按自己环境改）
├── train.sh           训练命令（可复现）
├── export_onnx.sh     导出 ONNX
└── results/           本次训练的全部产物
    ├── results.png / results.csv / args.yaml
    ├── BoxPR_curve.png / PosePR_curve.png
    ├── confusion_matrix_normalized.png / labels.jpg
    ├── train_batch0.jpg
    ├── val_batch0_labels.jpg / val_batch0_pred.jpg
    └── classes.txt
```

**权重文件（`best.pt` / `best.onnx`）未入库**（体积原因）。复现方式：按本目录的命令重新训练，或见根目录 README "准备模型"。
