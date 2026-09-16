#!/usr/bin/env bash
# 从零训练 YOLO26-pose 装甲板关键点模型
#
# 环境：Ubuntu 22.04 + Python 3.10 + ultralytics 8.4.142 + torch + CUDA
# 实测：RTX 4060 Laptop 8G，100 epoch 约 68 分钟
#
# 用法：把 dataset.yaml 里的 path 改成自己的数据集目录，然后
#   bash train.sh

set -e

DATA=dataset.yaml
NAME=armor_yolo26

# 确认用的是结构文件（.yaml）而不是权重（.pt）—— 这样才是从零训练
yolo pose train \
  model=yolo26-pose.yaml \
  data="${DATA}" \
  epochs=100 \
  imgsz=640 \
  batch=-1 \
  device=0 \
  workers=8 \
  name="${NAME}"

# 训练完成后：
#   产物在 runs/pose/${NAME}/
#   权重在 runs/pose/${NAME}/weights/best.pt
#   逐轮指标在 runs/pose/${NAME}/results.csv
#
# ⭐ 自检"是不是从零训练"：看 results.csv 第 1 行的 mAP50
#    接近 0（本次为 0.00084）→ 从随机初始化开始
#    明显偏高（比如 >0.1）→ 加载了预训练权重
#
# 显存不够时把 batch=-1 改成 batch=16 或更小。
