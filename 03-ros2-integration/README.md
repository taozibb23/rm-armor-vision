# 03 ROS2 集成：把 YOLO 推理包成节点

> 这一节说明**哪个是主程序**、各个文件干什么、以及**怎么在自己的机器上构建**。

---

## 1. 主程序

**`src/armor_yolo_node.cpp`** ← **这是本项目的主程序**

它做的事：

```
读视频/相机帧
  → 调用 onnxruntime 跑 YOLO26-pose 推理
  → 手写解码 (1,26,8400) → IoU → NMS
  → 在帧上画框和角点
  → 用 cv_bridge 转成 sensor_msgs/Image
  → 发布到 /armor_image
```

**验证方式**：

```bash
ros2 run armor_pkg armor_yolo_node
ros2 topic hz /armor_image        # 纯转发约 20Hz；接入 CPU 推理后降到几 Hz
rviz2                             # Add → By topic → /armor_image → Image
```

---

## 2. 各文件说明（别被文件数量迷惑）

| 文件 | 编译目标 | 是什么 |
|---|---|---|
| **`src/armor_yolo_node.cpp`** | `armor_yolo_node` | ⭐ **主程序**：YOLO ONNX 推理 + 发布带框图像 |
| `src/armor_algo.cpp` / `.hpp` | `vision_node`、`image_pub` | 早期**传统视觉**算法（HSV 阈值 + 灯条配对），是 YOLO 之前的路子 |
| `src/armor_type.hpp` | （头文件） | 装甲板数据结构定义 |
| `src/vision_node.cpp` | `vision_node` | 传统视觉节点（用 `armor_algo`） |
| `src/image_pub.cpp` | `image_pub` | 发布图像的节点（用 `armor_algo`） |
| `src/publisher.cpp` / `subscriber.cpp` / `publisher_coord.cpp` | 同名目标 | **ROS2 入门练习**（std_msgs 发布/订阅、自定义坐标消息）——保留作学习记录 |
| `src/cam_test.cpp` | 未编译 | 早期摄像头采集测试代码 |
| `src/cam_test` | ❌ | ~~编译产物，已从仓库移除~~ |

**从 `vision_node` → `armor_yolo_node` 的演进**是本仓库的一条暗线：
先用传统视觉（HSV + 几何配对）做，遇到鲁棒性瓶颈后，换成自己训练的深度学习模型。
两条路子都留在仓库里，可以对比。

---

## 3. ⚠️ 构建前必须改的地方

**`CMakeLists.txt` 里有写死的绝对路径**（原作者机器的路径），你 clone 下来必须改成自己的：

```cmake
add_executable(armor_yolo_node src/armor_yolo_node.cpp)
set_target_properties(armor_yolo_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "/home/user/yolo_env/lib/python3.10/site-packages/onnxruntime/capi")   # ← 改成你的
target_include_directories(armor_yolo_node PRIVATE
  /home/user/vision_learn/onnxruntime_include)                                          # ← 改成你的
target_link_libraries(armor_yolo_node ${OpenCV_LIBS}
  /home/user/yolo_env/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2)  # ← 改成你的
```

**三处都要改**：`INSTALL_RPATH`、头文件目录、`.so` 路径。

**更推荐的做法**（避免写死）：用环境变量或 `find_library`：

```cmake
# 用 -DONNXRUNTIME_DIR=/your/path 传入
set(ONNXRUNTIME_DIR "" CACHE PATH "onnxruntime 目录")
target_include_directories(armor_yolo_node PRIVATE ${ONNXRUNTIME_DIR}/include)
target_link_libraries(armor_yolo_node ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
set_target_properties(armor_yolo_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
```

---

## 4. 构建与运行

```bash
# 把这个包放进你的 ROS2 workspace
cp -r armor_pkg ~/ros2_ws/src/
cd ~/ros2_ws

# 改完 CMakeLists 里的三处路径后再构建
colcon build --packages-select armor_pkg

source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run armor_pkg armor_yolo_node
```

---

## 5. 踩坑（详见根目录 README 和 `docs/`）

### 5.1 `libonnxruntime.so.1: cannot open shared object file`

**编译期能找到 ≠ 运行期能找到。**

- 原因：colcon 产物的**运行期搜索路径（rpath）**里没有 onnxruntime 的目录；
  且 onnxruntime 的 SONAME 是"小名" `libonnxruntime.so.1`，而你链接的是带版本号的全名
- 定位：`ldd <可执行文件> | grep onnx`、`readelf -d <可执行文件> | head`
- 解法（三选一）：
  1. **CMake 里设 `INSTALL_RPATH`**（推荐，一劳永逸）
  2. 补软链：`ln -s libonnxruntime.so.1.23.2 libonnxruntime.so.1`
  3. 临时：`export LD_LIBRARY_PATH=/path/to/onnxruntime:$LD_LIBRARY_PATH`

### 5.2 推理把帧率拖到几 Hz

`ros2 topic hz` 从 **20Hz** 掉到 **几 Hz** —— 瓶颈是 **CPU 推理**，不是采集也不是发布。

**定位方法**：在采集、推理、发布三段各加计时探针，先确认是哪一段慢。

**优化方向**：降输入尺寸 / 跳帧 / 量化 FP16·INT8 / GPU（TensorRT）/ 多线程解耦。
