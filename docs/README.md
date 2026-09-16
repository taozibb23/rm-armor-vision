# docs：三篇配套博客（CSDN 已发布）

本目录是三篇文章的**全文存档**。访问量更好的版本在 CSDN：

| # | 文章 | 对应仓库环节 | CSDN | 数据 |
|---|---|---|---|---|
| 01 | 从 8400 张卡片到画框：用 C++ 跑通 YOLO 姿态模型 | `02-onnx-inference/` | [链接](https://blog.csdn.net/2604_96052374/article/details/164992742) | 395 阅读 · 6 赞 |
| 02 | 训练一个关键点模型（You Only Look Once）YOLO | `01-yolo-training/` | [链接](https://blog.csdn.net/2604_96052374/article/details/165122893) | 330 阅读 · 3 赞 |
| 03 | YOLO 装甲板检测接入 ROS2 | `03-ros2-integration/` | [链接](https://blog.csdn.net/2604_96052374/article/details/165356050) | 280 阅读 · 3 赞 |

## 这三篇分别讲什么

**01 —— C++ 推理全流程**
把训练好的 YOLO26-pose 装甲板模型导出 ONNX，用 C++ + onnxruntime 实现"预处理 → 推理 → 解码 → 去重 → 画框"全流程。
**重点不是"跑通了"，而是把每个中间步骤的坑都记下来了**（预处理一致性、8400 个候选怎么读、IoU 与 NMS 为什么这么写）。

**02 —— 训练**
从数据集配置到 100 epoch 训练，讲清"想自己训一个关键点模型该从哪下手"。

**03 —— ROS2 集成**
把单机 C++ 程序包成 ROS2 节点，用 `cv_bridge` 发布 `sensor_msgs/Image`，并在 RViz2 里可视化。
含 `libonnxruntime.so.1` 找不到的完整排查过程（SONAME 小名 + rpath）。

## 为什么把踩坑写在博客里

这三篇的共同特点是：**记的都是"做错了 → 怎么定位 → 怎么改"**，而不是"某函数怎么用"。
前者是工程能力，后者查文档就有。
