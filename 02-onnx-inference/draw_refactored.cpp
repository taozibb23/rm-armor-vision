// draw_refactored.cpp — YOLO26-pose 装甲板检测：C++ onnxruntime 完整推理管线（重构版）
//
// 设计目标：每个函数只干一件事，变量各有归属，magic number 全部命名
// 数据流：原图 → makeLetterbox(灰边画布) → toTensor(1x3x640x640) → 模型推理
//         → decode(8400 候选) → 排序 → nmsTopK(每簇留K个) → draw(画框)
//
// 编译（在 ~/rm-study/04-onnx-inference 下）：
//   g++ -std=c++17 draw_refactored.cpp \
//       -I /home/user/vision_learn/onnxruntime_include \
//       -L /home/user/yolo_env/lib/python3.10/site-packages/onnxruntime/capi \
//       -l:libonnxruntime.so.1.23.2 \
//       -Wl,-rpath,/home/user/yolo_env/lib/python3.10/site-packages/onnxruntime/capi \
//       $(pkg-config --cflags --libs opencv4) -o draw_refactored
// 运行：
//   ./draw_refactored <best.onnx> <图片>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdlib>

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
constexpr int   kKeepTop      = 2;                          // 每簇最多保留几个候选   // 每簇留 2 条：簇首几何不合格时第二候选可顶替 实测（500 帧测试片段）：救回次数 = 0 即簇首基本都合格；保留作为低成本保险
constexpr float kPadValue     = 114.f;                      // letterbox 灰边值（必须与训练一致）
constexpr float kNormScale    = 255.0f;                     // 像素归一化除数

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

// ============ 5) NMS ============
// 找簇的归属
std::vector<std::vector<int>> nms(const std::vector<Det>& dets) {
    std::vector<std::vector<int>> kept;
    for(size_t i = 0; i < dets.size(); ++i){//外层的i 遍历所有
        int belong = -1; //说明没有归属
        for(int g = 0; g < (int)kept.size(); ++g){//遍历簇
            if(iou(dets[i],dets[kept[g][0]]) > kIouThr){//和这个簇做比较
                belong = g;
                break;
            }
        }
    if(belong >= 0){//前面找到了簇归属
        if((int)kept[belong].size() < kKeepTop){
            kept[belong].push_back((int)i);
        }
    }else{
        kept.push_back({(int)i});
    }    
    }
    return kept;
}
// ============ 6) 画框：候选 → 可视化 ============
void draw(cv::Mat& canvas, const std::vector<Det>& dets,
          const std::vector<int>& kept) {
    for (int idx : kept) {
        const Det& d = dets[idx];
        const cv::Rect r(cvRound(d.cx - d.w / 2), cvRound(d.cy - d.h / 2),
                         cvRound(d.w), cvRound(d.h));
        cv::rectangle(canvas, r, cv::Scalar(0, 255, 0), 2);              // 绿框
        cv::putText(canvas, kClassNames[d.cls], cv::Point(r.x, r.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        std::vector<cv::Point> poly;                                     // 4 角点连线
        for (int k = 0; k < kKptRows; k += 2) {
            poly.push_back(cv::Point(cvRound(d.kpts[k]), cvRound(d.kpts[k + 1])));
        }
        cv::polylines(canvas, poly, true, cv::Scalar(0, 0, 255), 2);     // 红折线
    }
}
/// @brief 计算方法
/// @param a 点
/// @param b 点
/// @return a.x * b.y - a.y * b.x;
float cross(cv::Point2f a,cv::Point2f b)
{
    return a.x * b.y - a.y * b.x;
}


// =============7 图片的推理 ======================
void processFrame(cv::Mat &img, Ort::Session &session, std::ofstream &f, int frameIndx)
{   

    //=============处理逻辑全流程============
    float scale = 1.f;
    cv::Mat canvas = makeLetterbox(img, scale);
    std::vector<float> input = toTensor(canvas);

    std::array<int64_t, 4> inShape{1, 3, kInputSize, kInputSize};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(
        mem, input.data(), input.size(), inShape.data(), inShape.size());

    const char* inNames[]  = {"images"};
    const char* outNames[] = {"output0"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, inNames, &tensor, 1,
                               outNames, 1);

    const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* p = outputs[0].GetTensorData<float>();
    const int64_t numCandidates = shape[2];              // 8400
    std::vector<Det> dets = decode(p, numCandidates);
    std::sort(dets.begin(), dets.end(),
              [](const Det& a, const Det& b) { return a.conf > b.conf; });
    std::vector<std::vector<int>> kept = nms(dets);
    std::vector<int> finalKept;
    for(auto g : kept){//kept里面选出簇也就是外层
        for (int idx : g){//簇里面再次选内层的
        auto left_top = cv::Point2f(dets[idx].kpts[0]/scale, dets[idx].kpts[1]/scale);
        auto left_bottom = cv::Point2f(dets[idx].kpts[2]/scale, dets[idx].kpts[3]/scale);
        auto right_bottom = cv::Point2f(dets[idx].kpts[4]/scale, dets[idx].kpts[5]/scale);
        auto right_top = cv::Point2f(dets[idx].kpts[6]/scale, dets[idx].kpts[7]/scale);
        auto cx = (dets[idx].kpts[0] + dets[idx].kpts[2] + dets[idx].kpts[4] + dets[idx].kpts[6]) / 4 / scale;
        auto cy = (dets[idx].kpts[1] + dets[idx].kpts[3] + dets[idx].kpts[5] + dets[idx].kpts[7]) / 4 / scale;
        auto dx = left_bottom.x - left_top.x;
        auto dy = left_bottom.y - left_top.y;
        auto angle_rad = atan2(dx, dy);
        auto angle_deg = angle_rad * 180 / CV_PI;
            
         //=====================opencv===============
         //二维差积   判断同号
        auto P0 = left_top;
        auto P1 = left_bottom;
        auto P2 = right_bottom;
        auto P3 = right_top;
        //计算突四边形
        auto c0 = cross(P1-P0, P2-P1);
        auto c1 = cross(P2-P1, P3-P2);
        auto c2 = cross(P3-P2, P0-P3);
        auto c3 = cross(P0-P3, P1-P0);
        bool allSameSign = 
        (c0 > 0 && c1 > 0 && c2 > 0 && c3 > 0 ) ||
        (c0 < 0 && c1 < 0 && c2 < 0 && c3 < 0 );
        if (!allSameSign)continue;//不用break   break会把后面候选丢掉
        //对边边长相近
        auto d01 = cv::norm(P1-P0);
        auto d23 = cv::norm(P2-P3);
        auto d30 = cv::norm(P3-P0);
        auto d12 = cv::norm(P1-P2);
        const double r1 = std::abs(d01-d23) / std::max(d01, d23) ;
        const double r2 = std::abs(d30-d12) / std::max(d30, d12) ;
        if(!(r1 < 0.3 && r2 < 0.3)) continue;
        //这部分是几何检查的 候选conf是经过排序后的 后续可以 加上原来的rm_project

        f << frameIndx << "," << idx << "," << kClassNames[dets[idx].cls] << "," << dets[idx].conf << "," << cx << "," << cy << "," << angle_deg << "\n";
        finalKept.push_back(idx);
        break;
        }
    }
    draw(canvas, dets, finalKept);

}
// ============ main：只做编排，一眼看清整条管线 ============
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "用法: ./draw_refactored <模型.onnx> <图片>\n";
        return 1;
    }
    int imgFrame = 0;
    int frameidx = 0;
    cv::Mat videoframe;
    std::ofstream f("angles.csv");
    f << "frame,det_idx,cls,conf,cx,cy,angle_deg\n";
    // 1. 打开模型
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "armor");
    Ort::Session session(env, argv[1], Ort::SessionOptions(nullptr));

    // 2. 读图 + 预处理
    cv::Mat img = cv::imread(argv[2]);
    if(!img.empty())
    {
        processFrame(img, session, f, imgFrame);
        f.close();
    }
    if (img.empty()) 
    { 
        std::cout << "参数给的是视频流"<<std::endl; 
        cv::VideoCapture cap(argv[2]);
        double fps = cap.get(cv::CAP_PROP_FPS);
        while(true)
        {   
            cap >> videoframe;
            if(videoframe.empty()) break;
            processFrame(videoframe, session, f, frameidx);
            ++frameidx;
        }
        cap.release();
        f.close();
    }
    return 0;
}
