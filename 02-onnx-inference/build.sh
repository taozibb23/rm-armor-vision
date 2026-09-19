#!/usr/bin/env bash
# 编译 draw_refactored.cpp
#   用法：bash build.sh
#
# 换机器时改这三行：
ORT_INC=/home/user/vision_learn/onnxruntime_include
ORT_LIB=/home/user/yolo_env/lib/python3.10/site-packages/onnxruntime/capi
ORT_SO=libonnxruntime.so.1.23.2

cd "$(dirname "$0")" || exit 1

# 先删掉旧的可执行文件 —— 这样"编译失败"不会被旧文件掩盖
rm -f draw_refactored

# 不加管道，让编译器的报错原样打印，退出码也保留
if g++ -std=c++17 draw_refactored.cpp \
      -I "${ORT_INC}" \
      -L "${ORT_LIB}" \
      -l:"${ORT_SO}" \
      -Wl,-rpath,"${ORT_LIB}" \
      $(pkg-config --cflags --libs opencv4) \
      -o draw_refactored ; then
    echo "✅ 编译成功 → draw_refactored"
else
    echo "❌ 编译失败（看上面的 error: 行）"
    exit 1
fi
