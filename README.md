# RM 装甲板视觉：从 YOLO 训练到 C++ / ROS2 部署

> **一句话**：自己从零训练 YOLO26-pose 装甲板关键点模型 → 导出 ONNX → 用 **C++ + onnxruntime 手写完整推理管线**（预处理 / 解码 / NMS / 坐标还原）→ 接进 **ROS2** 发布带框图像话题。

## 30 秒看点

| 你可能想知道 | 在哪看 |
|---|---|
| 模型是自己训练的吗？ | [`01-yolo-training/`](01-yolo-training/) —— 训练命令、超参数、逐 epoch 指标、**从零训练的证据** |
| 推理代码是手写的还是调包？ | [`02-onnx-inference/`](02-onnx-inference/) —— letterbox、8400 候选解码、IoU、NMS 全是自己写的 |
| 能接进真实系统吗？ | [`03-ros2-integration/`](03-ros2-integration/) —— ROS2 节点 + `cv_bridge` 发布 `sensor_msgs/Image` |
| 效果到底怎么样？ | 下面「效果与数据」+ [`results.csv`](01-yolo-training/results/results.csv)（逐轮原始数字） |
| 遇到过什么坑？ | 下面「已知问题」+ [`docs/`](docs/)（三篇踩坑博客） |

---

## 效果与数据

| 项目 | 结果 |
|---|---|
| 训练数据 | XJTLU 2023 Keypoints（6732 张：**train 4224 / val 2508**；14 类；4 角点） |
| 训练方式 | **从零训练**（随机初始化 + 100 epoch，约 68 分钟） |
| **检测框** | Precision 0.850 / Recall 0.760 / **mAP50 0.842** / **mAP50-95 0.559** |
| **关键点（角点）** | Precision 0.744 / Recall 0.662 / **mAP50 0.675** / **mAP50-95 0.610** |
| 部署形态 | C++ + onnxruntime，手写 letterbox 预处理 / 8400 候选解码 / IoU / NMS / 坐标还原 |
| ROS2 集成 | 节点发布 `sensor_msgs/Image`（带框）→ RViz2 可视化 |
| 性能实测 | 纯转发 ≈ **20Hz**；接入 CPU 推理后降到 **几 Hz**（瓶颈与优化方向见下文） |

### 训练曲线

![训练曲线](01-yolo-training/results/results.png)

### 验证集预测效果（左：真值 / 右：预测）

| 真值 | 预测 |
|---|---|
| ![labels](01-yolo-training/results/val_batch0_labels.jpg) | ![pred](01-yolo-training/results/val_batch0_pred.jpg) |

**逐类结果里发现的问题**：`R5` 检测 mAP50 有 0.897，但**角点 mAP50 只有 0.329** —— 说明是这类样本的关键点标注有问题，**下一步该去查标注而不是调超参**。完整表见 [`01-yolo-training/README.md`](01-yolo-training/README.md#52-逐类结果这是最能看出问题的一张表)。

---

## 目录结构

```
.
├── 01-yolo-training/      训练：数据配置、训练命令、100 epoch 全部产物与指标
│   ├── dataset.yaml       数据集配置（14 类 / 4 关键点）
│   ├── train.sh           训练命令（可复现）
│   ├── export_onnx.sh     导出 ONNX
│   └── results/           训练曲线、PR 曲线、混淆矩阵、args.yaml、results.csv
├── 02-onnx-inference/     C++ 推理：预处理 → 解码 → NMS → 画框 → 视频输出
├── 03-ros2-integration/   ROS2 节点：发布带框图像话题 + CMake 配置
└── docs/                  三篇踩坑博客（SONAME/rpath、预处理一致性、ROS2 集成）
```

---

## 环境

- **操作系统**：Ubuntu 22.04
- **推理侧**：OpenCV 4.5.4，onnxruntime 1.23.2（CPU）
- **机器人侧**：ROS2 Humble
- **训练侧**：Python 3.10.12 + ultralytics 8.4.142 + PyTorch，RTX 4060 Laptop（8G）

---

## 快速开始

### 0. 准备模型

把训练导出的 `best.onnx` 放到 `02-onnx-inference/weights/`。

> 模型文件（`.onnx` / `.pt`）因体积原因未入库。可以按 [`01-yolo-training/train.sh`](01-yolo-training/train.sh) 自己训练 + [`export_onnx.sh`](01-yolo-training/export_onnx.sh) 导出。

### 1. C++ 单机推理（处理视频）

```bash
cd 02-onnx-inference
g++ -std=c++17 draw_refactored.cpp -o draw_refactored \
    -I /path/to/onnxruntime_include \
    -L /path/to/onnxruntime/lib -l:libonnxruntime.so.1.23.2 \
    -Wl,-rpath,/path/to/onnxruntime/lib \
    $(pkg-config --cflags --libs opencv4)

./draw_refactored weights/best.onnx test.mp4     # 输出 out.mp4
```

### 2. ROS2 集成

```bash
# 把 03-ros2-integration/armor_pkg 放进你的 ROS2 workspace
cd ~/ros2_ws
colcon build --packages-select armor_pkg
source /opt/ros/humble/setup.bash && source install/setup.bash

ros2 run armor_pkg armor_yolo_node
ros2 topic hz /armor_image      # 另开终端：看频率
rviz2                           # Add → By topic → /armor_image → Image
```

---

## 已知问题与排查（详见 [`docs/`](docs/)）

| 现象 | 原因 | 解法 |
|---|---|---|
| `libonnxruntime.so.1: cannot open shared object file` | 动态库 **SONAME "小名"** + 运行期搜索路径缺失（**编译期能找到 ≠ 运行期能找到**） | `ln -s` 补小名软链；CMake 加 `INSTALL_RPATH`；或临时 `LD_LIBRARY_PATH` |
| 检测结果整体偏移 / 精度异常 | **预处理与训练不一致** | 逐项检查：灰边值 114、BGR→RGB、÷255、HWC→CHW |
| 频率只有几 Hz | **CPU 推理是瓶颈**（不是采集、不是发布） | 优化方向：降输入尺寸 / 跳帧 / 量化 FP16·INT8 / GPU(TensorRT) / 多线程解耦 |

**定位方法**：在采集、处理、发布三段各加计时探针，先确认瓶颈在哪一段，再优化——**不要凭感觉优化**。

---

## 相关链接

- 博客（三篇）
  1. 从 8400 张卡片到画出一个框（C++ 推理全流程）
  2. 从零训练 YOLO26-pose 到 C++ 部署
  3. 把 YOLO 装甲板检测接进 ROS2
- 传统视觉装甲板检测项目：`RM_project`（HSV 阈值 + 灯条几何配对 + 串口通信）
- 作者主页：<https://blog.csdn.net/2604_96052374>

---

## 说明

本项目为个人学习与 RoboMaster 战队考核用，欢迎交流指正。
数据集 **XJTLU 2023 Keypoints** 为公开数据集，引用请注明来源。
