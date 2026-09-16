#!/usr/bin/env bash
# 导出 ONNX（给 C++ 推理用）
#
# 用法：bash export_onnx.sh
# 产物：runs/pose/armor_yolo26/weights/best.onnx（约 10.2MB）

set -e

PT=runs/pose/armor_yolo26/weights/best.pt

yolo export \
  model="${PT}" \
  format=onnx \
  imgsz=640 \
  opset=12 \
  simplify=True

# 导出的模型：
#   输入  (1, 3, 640, 640)   NCHW，RGB，值域 0~1
#   输出  (1, 26, 8400)
#
# 输出 26 行的含义（C++ 侧解码要用）：
#   行 0~3   框：cx, cy, w, h
#   行 4~17  14 个类别分数        ← 类号 = 行号 − 4
#   行 18~25 4 个角点的 (x, y)    ← 每 2 行一个点
#   8400     候选数量 = 80×80 + 40×40 + 20×20 = 6400 + 1600 + 400
#
# 验证输出维度：
#   python -c "import onnxruntime as ort; s=ort.InferenceSession('runs/pose/armor_yolo26/weights/best.onnx'); print([o.shape for o in s.get_outputs()])"
