# 03 ROS2 集成

本目录把 YOLO 推理包成 ROS2 节点，发布带框图像话题，供 RViz2 或下游节点订阅。

---

## 1. 主程序

**`src/armor_yolo_node.cpp`**

数据流：

```
读取视频帧 / 相机帧
  → onnxruntime 执行 YOLO26-pose 推理
  → 解码输出 (1, 26, 8400) → IoU 过滤 → NMS
  → 绘制检测框与角点
  → cv_bridge 转为 sensor_msgs/Image
  → 发布到 /armor_image
```

**运行验证**：

```bash
ros2 run armor_pkg armor_yolo_node
ros2 topic hz /armor_image        # 纯转发约 20Hz；接入 CPU 推理后降至几 Hz
rviz2                             # Add → By topic → /armor_image → Image
```

---

## 2. 各源文件说明

| 文件 | 编译目标 | 说明 |
|---|---|---|
| **`src/armor_yolo_node.cpp`** | `armor_yolo_node` | **主程序**：YOLO ONNX 推理 + 发布带框图像 |
| `src/armor_algo.cpp` / `.hpp` | `vision_node`、`image_pub` | 早期**传统视觉**算法（HSV 阈值 + 灯条几何配对） |
| `src/armor_type.hpp` | （头文件） | 装甲板数据结构定义 |
| `src/vision_node.cpp` | `vision_node` | 传统视觉节点（依赖 `armor_algo`） |
| `src/image_pub.cpp` | `image_pub` | 图像发布节点（依赖 `armor_algo`） |
| `src/publisher.cpp` / `subscriber.cpp` / `publisher_coord.cpp` | 同名目标 | ROS2 发布/订阅基础练习（std_msgs 与自定义坐标消息），作为学习记录保留 |
| `src/cam_test.cpp` | 未编译 | 早期摄像头采集测试代码 |

**两条技术路线的对比**：`vision_node`（传统视觉：HSV + 几何配对）与 `armor_yolo_node`（深度学习）解决的是同一个问题。
前者在复杂光照与遮挡下鲁棒性不足，因此转为自训练模型；两条路线的代码均保留，便于对照。

---

## 3. 构建前的必要修改

**`CMakeLists.txt` 中包含开发机上的绝对路径**，在其他机器上构建前必须替换：

```cmake
add_executable(armor_yolo_node src/armor_yolo_node.cpp)
set_target_properties(armor_yolo_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "/home/user/yolo_env/lib/python3.10/site-packages/onnxruntime/capi")   # ← 本机 onnxruntime 库目录
target_include_directories(armor_yolo_node PRIVATE
  /home/user/vision_learn/onnxruntime_include)                                          # ← 本机 onnxruntime 头文件目录
target_link_libraries(armor_yolo_node ${OpenCV_LIBS}
  /home/user/yolo_env/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2)  # ← 本机 .so 路径
```

共三处：`INSTALL_RPATH`、头文件目录、`.so` 路径。

**更推荐通过变量传入，避免写死**：

```cmake
set(ONNXRUNTIME_DIR "" CACHE PATH "onnxruntime 安装目录")
target_include_directories(armor_yolo_node PRIVATE ${ONNXRUNTIME_DIR}/include)
target_link_libraries(armor_yolo_node ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
set_target_properties(armor_yolo_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
```

构建时传入：`colcon build --cmake-args -DONNXRUNTIME_DIR=/path/to/onnxruntime`

---

## 4. 构建与运行

```bash
cp -r armor_pkg ~/ros2_ws/src/
cd ~/ros2_ws

# 先按第 3 节修改 CMakeLists.txt 中的路径
colcon build --packages-select armor_pkg

source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run armor_pkg armor_yolo_node
```

---

## 5. 遇到的问题与排查

### 5.1 `libonnxruntime.so.1: cannot open shared object file`

**编译期能找到 ≠ 运行期能找到。**

- **原因**：colcon 产物的运行期搜索路径（rpath）中不含 onnxruntime 目录；同时 onnxruntime 的 SONAME 是 `libonnxruntime.so.1`（"小名"），而链接时使用的是带完整版本号的文件名
- **定位**：`ldd <可执行文件> | grep onnx`、`readelf -d <可执行文件> | head`
- **处理**（三选一）：
  1. 在 CMake 中设置 `INSTALL_RPATH`（推荐，一劳永逸）
  2. 补建软链：`ln -s libonnxruntime.so.1.23.2 libonnxruntime.so.1`
  3. 临时方案：`export LD_LIBRARY_PATH=/path/to/onnxruntime:$LD_LIBRARY_PATH`

### 5.2 接入推理后帧率下降

`ros2 topic hz` 由 **20Hz** 降至 **几 Hz**。瓶颈在 **CPU 推理**，而非采集或发布环节。

**定位方法**：在采集、推理、发布三段分别加计时探针，先确认耗时集中在哪一段。

**后续优化方向**：降低输入尺寸、跳帧、量化（FP16 / INT8）、改用 GPU（TensorRT）、多线程解耦。
