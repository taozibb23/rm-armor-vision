#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdio>


int main(int argc, char** argv) {
    if (argc < 3) { std::cout << "用法: ./infer 模型 图片\n"; return 1; }
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "infer");  //类似一个管家 管理运行环境  
    Ort::Session session(env, argv[1], Ort::SessionOptions(nullptr)); //env的管家把模型文件打开
    cv::Mat img = cv::imread(argv[2]);//输入
    if (img.empty()) { std::cout << "图片打不开\n"; return 1; }

    const int S = 640;//后面 缩放等要用到
    // 1) letterbox：等比例缩放 + 灰边补成 640x640
    float scale = std::min((float)S / img.cols, (float)S / img.rows);  //换算比例
    cv::Mat resized;//接受 变化的mat
    cv::resize(img, resized, cv::Size(cvRound(img.cols*scale), cvRound(img.rows*scale)));//缩放比例
    cv::Mat canvas(S, S, CV_8UC3, cv::Scalar(114,114,114));//不够的时候用于填充的比例 数值与训练的时候一致也是114
    resized.copyTo(canvas(cv::Rect(0, 0, resized.cols, resized.rows)));//填充画布与缩放比例后的图融合
    // 2) BGR -> RGB
    cv::Mat rgb; cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);//bgr转rgb
    // 3) float /255 / HWC->CHW，装入 1x3x640x640
    std::vector<float> input(3*S*S);
    for (int c=0;c<3;c++)
        for (int y=0;y<S;y++)
            for (int x=0;x<S;x++)
                input[c*S*S + y*S + x] = rgb.at<cv::Vec3b>(y,x)[c] / 255.0f;//拆手册因为矩阵的格式被转化成行的格式对于 chw需要吧红绿蓝分开 循环3层拆开并且整理

    std::array<int64_t,4> inShape{1,3,S,S};//shape描述图片的结构 1张图片 3个通道rgb 长宽都是640
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);//说明数据储存在cpu的内存里面
    auto tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                  inShape.data(), inShape.size());
    //把数字和描述的标签打包在一起input是啥？ 和形状打包成一起
    const char* inName="images"; const char* outName="output0";//输入口输出口的名字
    auto out = session.Run(Ort::RunOptions{nullptr}, &inName, &tensor, 1, &outName, 1);

    auto shp = out[0].GetTensorTypeAndShapeInfo().GetShape();//获取类型的信息还有获取形状信息
    std::cout << "输出形状: "; for (auto s:shp) std::cout<<s<<" "; std::cout<<"\n";

    const float* p = out[0].GetTensorData<float>();
    int64_t total=shp[2], ch=shp[1], high=0;
    for (int64_t i=0;i<total;i++){           // 遍历 8400 个候选
        float best=0;
        for (int k=4;k<18;k++) best=std::max(best, p[k*total+i]);  // 行4~17是14类分(假设)
        if (best>0.5f) high++;
    }
    std::cout << "信心>0.5 的候选数: " << high << "\n";
    //找出最高分数候选
    int bestK = -1; int64_t bestI = 0; float bestV = 0;
    for (int64_t i = 0; i <total; i++){
        for(int k = 4; k < 18; k++){
            if(p[k*total+i] > bestV){bestV = p[k*total+i]; bestK = k; bestI = i; }
        }
    }
    printf("最高分候选: 位置i=%ld 类别行k=%d 分=%.3f\n", bestI, bestK, bestV);
    printf("box 前4行: cx=%.1f cy=%.1f w=%.1f h=%.1f\n",
        p[0*total+bestI], p[1*total+bestI], p[2*total+bestI], p[3*total+bestI]);
    printf("14类分(4-17行):");
    for (int k = 4; k < 18;k++)printf("%.2f", p[k*total+bestI]);
    printf("\n角点(行18~25): ");
    for (int k = 18; k < 26; k++) printf("%.1f ", p[k*total+bestI]);
    printf("\n");

    return 0;
}