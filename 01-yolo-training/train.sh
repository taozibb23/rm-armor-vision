#!/usr/bin/env bash
# 训练 YOLO26-pose 装甲板关键点模型
#
# 环境：Ubuntu 22.04 + Python 3.10 + ultralytics 8.4.142 + torch + CUDA
# 实测：RTX 4060 Laptop 8G，100 epoch 约 68 分钟
#
# 用法：先将 dataset.yaml 中的 path 指向本地数据集目录，然后
#   bash train.sh

set -e

DATA=dataset.yaml
NAME=armor_yolo26

# model 使用结构文件（.yaml）而非权重文件（.pt），从随机初始化开始训练
yolo pose train \
  model=yolo26-pose.yaml \
  data="${DATA}" \
  epochs=100 \
  imgsz=640 \
  batch=-1 \
  device=0 \
  workers=8 \
  name="${NAME}"

# 训练产物：
#   runs/pose/${NAME}/weights/best.pt      权重
#   runs/pose/${NAME}/results.csv          逐轮指标
#   runs/pose/${NAME}/results.png          训练曲线
#
# 判断本次是否从随机初始化开始：查看 results.csv 第 1 行的 mAP50。
#   接近 0（本次为 0.00084）      → 从随机初始化开始
#   明显偏高（例如 > 0.1）        → 加载了预训练权重
#
# 显存不足时将 batch=-1 改为 batch=16 或更小。
