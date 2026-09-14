# RM 装甲板视觉：从 YOLO 训练到 C++ / ROS2 部署

> 一句话：自己训练 YOLO26-pose 装甲板关键点模型 → 导出 ONNX → 用 C++ + onnxruntime 手写完整推理管线 → 接进 ROS2 发布带框图像话题。

## 效果与数据

| 项目 | 结果 |
|---|---|
| 训练数据 | XJTLU 2023 Keypoints（6732 张：train 4224 / val 2508；14 类；4 角点） |
| 模型指标 | 框 **mAP50 0.84** / mAP50-95 0.559；角点 mAP50 0.675 / **mAP50-95 0.61**（从零训练，100 epoch） |
| 部署形态 | C++ + onnxruntime，手写 letterbox 预处理 / 8400 候选解码 / NMS / 坐标还原 |
| ROS2 集成 | 节点发布 `sensor_msgs/Image`（带框）→ RViz2 可视化 |
| 性能实测 | 纯转发 ≈20Hz；接入 CPU 推理后降到几 Hz（瓶颈与优化方向见 `docs/`） |

（建议此处放两张图：`results.png` 训练曲线 + RViz2 带框截图 / out.mp4 演示）

## 目录结构

```
.
├── 01-yolo-training/      训练：数据准备、命令、日志解读
├── 02-onnx-inference/     C++ 推理：预处理→解码→NMS→画框→视频输出
├── 03-ros2-integration/   ROS2 节点：发布带框图像话题 + CMake 片段
└── docs/                  博客与踩坑记录（SONAME/rpath、预处理一致性等）
```

## 环境

- Ubuntu 22.04，OpenCV 4.5.4，onnxruntime 1.23.2（CPU）
- ROS2 Humble
- 训练侧：Python 3.10 + ultralytics 8.4.142 + PyTorch，RTX 4060 Laptop

## 快速开始

### 0. 准备模型
把训练导出的 `best.onnx` 放到 `02-onnx-inference/weights/`（模型文件较大，可用 Git LFS 或自行训练）。

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
cd ~/ros2_ws && colcon build --packages-select armor_pkg
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 run armor_pkg armor_yolo_node
ros2 topic hz /armor_image      # 另开终端
rviz2                          # Add → By topic → /armor_image → Image
```

## 已知问题与排查（详见 docs/）

| 现象 | 原因 | 解法 |
|---|---|---|
| `libonnxruntime.so.1: cannot open shared object file` | 动态库 SONAME "小名" + 运行期搜索路径缺失（**编译期能找到 ≠ 运行期能找到**） | `ln -s` 补小名软链；CMake 加 `INSTALL_RPATH`；或临时 `LD_LIBRARY_PATH` |
| 检测结果整体偏移 / 精度异常 | 预处理与训练不一致 | 检查灰边值 114、BGR→RGB、÷255、HWC→CHW |
| 频率只有几 Hz | CPU 推理是瓶颈 | 优化方向：降输入尺寸 / 跳帧 / 量化 FP16·INT8 / GPU(TensorRT) / 多线程 |

## 相关链接

- 博客（三篇）：① 从 8400 张卡片到画出一个框（C++ 推理全流程） ② 从零训练 YOLO26-pose 到 C++ 部署 ③ 把 YOLO 装甲板检测接进 ROS2
- 传统视觉装甲板检测项目：`RM_project`（HSV + 灯条几何配对 + 串口）
- 作者主页：https://blog.csdn.net/2604_96052374

## 说明

本项目为个人学习与 RM 战队考核用，欢迎交流指正。数据集 XJTLU 2023 Keypoints 为公开数据集；引用请注明来源。
