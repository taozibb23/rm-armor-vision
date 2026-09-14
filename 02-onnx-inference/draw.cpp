#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>

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
    int64_t total = shp[2];

    //解码 遍历8400个 得最高分数
    struct Det {float conf; int cls; float cx,cy,w,h; float pts[8];};
    std::vector<Det> dets;
    for (int64_t i =0; i < total; i++){
        float best = 0; int bestC = -1;
        for (int k = 4; k < 18; k++)
            if (p[k*total+i] > best){best = p[k*total+i]; bestC = k -4;}
        if(best > 0.25f){//conf度
            Det d; d.conf = best; d.cls = bestC;
            d.cx = p[0*total+i]; d.cy = p[1*total+i];
            d.w = p[2*total+i]; d.h = p[3*total+i];
            for (int j = 0; j < 8;j++)d.pts[j] = p[(18+j)*total+i];
            dets.push_back(d);
        }
    }
    std::cout<<"高分候选数:"<<dets.size() <<"\n";

    auto iou = [](const Det& a, const Det& b) -> float{
        float ax1=a.cx-a.w/2, ay1=a.cy-a.h/2, ax2=a.cx+a.w/2, ay2=a.cy+a.h/2;
        float bx1=b.cx-b.w/2, by1=b.cy-b.h/2, bx2=b.cx+b.w/2, by2=b.cy+b.h/2;
        float ix1=std::max(ax1, bx1), iy1=std::max(ay1, by1);
        float ix2=std::min(ax2,bx2), iy2=std::min(ay2,by2);
        float iw=std::max(0.f,ix2-ix1), ih=std::max(0.f,iy2-iy1);
        float inter=iw*ih, uni=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter;
        return uni>0 ? inter/uni : 0.f;
    };
    std::sort(dets.begin(), dets.end(),
            [](const Det& a, const Det& b){return a.conf > b.conf;});
    const float iouThr = 0.5f;
    const int keepTop =2;
    std::vector<int> kept;
    for (size_t i =0; i < dets.size(); i++){
        int sameCluster = 0;
        for(int idx : kept)
            if (iou(dets[i], dets[idx]) > iouThr)sameCluster++;
        if(sameCluster < keepTop) kept.push_back((int)i);
    }
    std::cout <<"NMS后保留的数:"<<kept.size()<<"个\n";

    for (int idx : kept) printf("保留 idx=%d conf=%.2f 类=%d\n", idx, dets[idx].conf, dets[idx].cls);

    for (size_t n = 0; n < dets.size(); n++)
        printf("[%zu]类=%d conf=%.2f 框=(%.0f,%.0f)%.0fx%.0f\n",
        n, dets[n].cls, dets[n].conf, dets[n].cx, dets[n].cy, dets[n].w, dets[n].h);
    // 画在letterbox画布上面
    const char* clsNames[14] = {"B1","B2","B3","B4","B5","BO","BS",
                                "R1","R2","R3","R4","R5","RO","RS"};
    cv::Mat show = canvas.clone();
    for (int idx : kept){
        Det& d = dets[idx];
        cv::Rect r(cvRound(d.cx - d.w/2),cvRound(d.cy - d.h/2),
                    cvRound(d.w),cvRound(d.h));
        cv::rectangle(show, r,cv::Scalar(0,255,0), 2);
        cv::putText(show, clsNames[d.cls],
                    cv::Point(r.x,r.y-5), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0,255,0), 2);
        std::vector<cv::Point> poly;
        for(int j =0; j < 8;j += 2)
            poly.push_back(cv::Point(cvRound(d.pts[j]), cvRound(d.pts[j+1])));
        cv::polylines(show, poly, true, cv::Scalar(0,0,255), 2);
    }
    cv::imwrite("draw_result.jpg", show);
    std::cout <<"已经保存 draw_result.jpg\n";

    return 0;
}