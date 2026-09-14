#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <vector>

// ============ 常量：把"魔法数字"命名，看名字就懂含义 ============
constexpr int   kInputSize    = 640;                        // 模型输入边长
constexpr int   kNumClasses   = 14;                         // 类别数 B1..BS / R1..RS
constexpr int   kNumKeypoints = 4;                          // 装甲板角点数
constexpr int   kBoxRows      = 4;                          // 输出行 0~3：cx,cy,w,h
constexpr int   kKptRows      = kNumKeypoints * 2;          // 输出行 18~25：4 点 ×(x,y)
constexpr int   kClassRow0    = kBoxRows;                   // 类别分行起点 = 4
constexpr int   kKptRow0      = kBoxRows + kNumClasses;     // 角点行起点   = 18
constexpr int   kNumRows      = kKptRow0 + kKptRows;        // 总行数 = 26
constexpr float kConfThr      = 0.25f;                      // 候选门槛（越低召回越高）
constexpr float kIouThr       = 0.5f;                       // 同簇判定：重合多少算同一目标
constexpr int   kKeepTop      = 2;                          // 每簇最多保留几个候选
constexpr float kPadValue     = 114.f;                      // letterbox 灰边值（必须与训练一致）
constexpr float kNormScale    = 255.0f;                     // 像素归一化除数

//这里的KNumClasses是对应上面的常量的
const char* kClassNames[kNumClasses] = {
    "B1", "B2", "B3", "B4", "B5", "BO", "BS",
    "R1", "R2", "R3", "R4", "R5", "RO", "RS"
};

// ============ 数据结构：一个候选框 ============
struct Det {
    float conf = 0.f;                                       // 置信度 = 14 类分里的最大值
    int   cls  = -1;                                        // 类别号（0~13）
    float cx = 0.f, cy = 0.f, w = 0.f, h = 0.f;             // 框（letterbox 空间的像素）
    float kpts[kKptRows] = {0.f};                           // 4 个角点 x,y（同空间）
};

// ============ 1) letterbox：等比缩放 + 灰边 → 640x640 画布 ============
// 预处理和画图都基于同一张画布，保证坐标一致
cv::Mat makeLetterbox(const cv::Mat& img, float& scaleOut) {
    const float scale = std::min((float)kInputSize / img.cols,
                                 (float)kInputSize / img.rows);
    scaleOut = scale;                                       // 还原坐标时要乘回去
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(cvRound(img.cols * scale),
                                      cvRound(img.rows * scale)));
    cv::Mat canvas(kInputSize, kInputSize, CV_8UC3,
                   cv::Scalar(kPadValue, kPadValue, kPadValue));
    // 贴在左上角 → 灰边只出现在右侧和底部（还账时按这个约定减偏移）
    resized.copyTo(canvas(cv::Rect(0, 0, resized.cols, resized.rows)));
    return canvas;
}

// ============ 2) 画布 → 模型输入张量 ============
// BGR→RGB（训练时是 RGB）、除以 255（归一化）、HWC→CHW（按颜色分册）
std::vector<float> toTensor(const cv::Mat& bgrCanvas) {
    cv::Mat rgb;
    cv::cvtColor(bgrCanvas, rgb, cv::COLOR_BGR2RGB);

    std::vector<float> input(3 * kInputSize * kInputSize);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < kInputSize; ++y) {
            for (int x = 0; x < kInputSize; ++x) {
                input[c * kInputSize * kInputSize + y * kInputSize + x] =
                    rgb.at<cv::Vec3b>(y, x)[c] / kNormScale;
            }
        }
    }
    return input;
}

// ============ 3) 解码：输出张量 → 候选列表 ============
// 输出布局 (1, 26, 8400)：行 0~3 框 / 行 4~17 类别分 / 行 18~25 角点
// 取第 k 行第 i 列 = p[k * total + i]（行优先摊平：先跳 k 整行，再走 i 个）
std::vector<Det> decode(const float* p, int64_t total) {
    std::vector<Det> dets;
    for (int64_t i = 0; i < total; ++i) {
        float best = 0.f;
        int   bestCls = -1;
        for (int k = 0; k < kNumClasses; ++k) {
            const float score = p[(kClassRow0 + k) * total + i];
            if (score > best) { best = score; bestCls = k; }  // 类号 = 行号 − kClassRow0
        }
        if (best < kConfThr) { continue; }                    // 低分候选直接丢

        Det d;
        d.conf = best;
        d.cls  = bestCls;
        d.cx   = p[0 * total + i];
        d.cy   = p[1 * total + i];
        d.w    = p[2 * total + i];
        d.h    = p[3 * total + i];
        for (int k = 0; k < kKptRows; ++k) {
            d.kpts[k] = p[(kKptRow0 + k) * total + i];
        }
        dets.push_back(d);
    }
    return dets;
}

// ============ 4) IoU：两个框的重合度（0~1）============
float iou(const Det& a, const Det& b) {
    const float ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
    const float ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
    const float bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
    const float bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;

    const float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);   // 交集左上
    const float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);   // 交集右下
    const float iw = std::max(0.f, ix2 - ix1);
    const float ih = std::max(0.f, iy2 - iy1);
    const float inter = iw * ih;
    const float uni = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    return (uni > 0.f) ? inter / uni : 0.f;
}

// ============ 5) NMS（宽松版）：每簇最多留 kKeepTop 个 ============
// 约定：dets 已按 conf 降序。数"已保留里同簇的数量"，没满 K 就留下
std::vector<int> nmsTopK(const std::vector<Det>& dets) {
    std::vector<int> kept;
    for (size_t i = 0; i < dets.size(); ++i) {
        int sameCluster = 0;
        for (int idx : kept) {
            if (iou(dets[i], dets[idx]) > kIouThr) { ++sameCluster; }
        }
        if (sameCluster < kKeepTop) { kept.push_back((int)i); }
    }
    return kept;
}

// ============ 6) 画框：候选 → 可视化 ============
void draw(cv::Mat& canvas, const std::vector<Det>& dets,
          const std::vector<int>& kept, float scale) {
    for (int idx : kept) {
        const Det& d = dets[idx];
        const float cx = d.cx / scale , cy = d.cy /scale;
        const float w  = d.w / scale, h = d.h /scale;
        const cv::Rect r(cvRound(cx - w / 2), cvRound(cy - h / 2),
                         cvRound(w), cvRound(h));
        cv::rectangle(canvas, r, cv::Scalar(0, 255, 0), 2);              // 绿框
        cv::putText(canvas, kClassNames[d.cls], cv::Point(r.x, r.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        std::vector<cv::Point> poly;                                     // 4 角点连线
        for (int k = 0; k < kKptRows; k += 2) {
            poly.push_back(cv::Point(cvRound(d.kpts[k] / scale), cvRound(d.kpts[k + 1] / scale)));
        }
        cv::polylines(canvas, poly, true, cv::Scalar(0, 0, 255), 2);     // 红折线
    }
}


int main(int argc, char** argv){
    rclcpp::init(argc,argv);
    auto node = rclcpp::Node::make_shared("armor_yolo_node");
    auto pub = node->create_publisher<sensor_msgs::msg::Image>(
        "/armor_image", 10);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "armor");
    Ort::Session session(env,
         "/home/user/vision_learn/视觉考核参考/runs/pose/armor_yolo26/weights/best.onnx",
        Ort::SessionOptions(nullptr));
    
        cv::VideoCapture cap("/home/user/vision_learn/视觉考核参考/test.mp4");
    if(!cap.isOpened()){
        RCLCPP_ERROR(node->get_logger(), "视频打不开");
        return 1;
    }
    rclcpp::WallRate rate(5);
    cv::Mat frame;
    while(rclcpp::ok() && cap.read(frame)){
        if(frame.empty()){break; }

        //预处理环节
        float scale = 1.f;
        cv::Mat canvas = makeLetterbox(frame, scale);
        std::vector<float> input = toTensor(canvas);
        //推理环节
        std::array<int64_t, 4> inShape{1, 3, kInputSize, kInputSize};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator ,OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(
            mem, input.data(), input.size(), inShape.data(), inShape.size()
        );
        const char* inNames[] = {"images"};
        const char* outNames[] = {"output0"};
        auto outputs = session.Run(Ort::RunOptions{nullptr}, inNames, &tensor, 1, outNames, 1);
        //解码和去重
        const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const float* p = outputs[0].GetTensorData<float>();
        std::vector<Det> dets = decode(p, shape[2]);
        std::sort(dets.begin(), dets.end(),
                    [](const Det& a, const Det& b){ return a.conf > b.conf;});
        std::vector<int> kept = nmsTopK(dets);
        //还原sclae
        draw(frame, dets, kept, scale);
        //发布带框图像

        cv_bridge::CvImage bridge;
        bridge.encoding = "bgr8";
        bridge.header.stamp = node->now();
        bridge.header.frame_id = "camera";
        bridge.image = frame;
        pub->publish(*bridge.toImageMsg());
        
        RCLCPP_INFO(node->get_logger(), "候选 %zu, 保留 %zu", dets.size(), kept.size());
        rate.sleep();
    }
    RCLCPP_INFO(node->get_logger(),"视频播放完");
    rclcpp::shutdown();
    return 0;
}