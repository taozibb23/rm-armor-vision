#include <opencv2/opencv.hpp>
#include <iostream>
int main() {
    cv::VideoCapture cap(1);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cv::Mat img;
    double t0 = cv::getTickCount();
    int frames = 0;
    while (frames < 100) {
        cap >> img;
        if (img.empty()) continue;
        frames++;
    }
    double t1 = cv::getTickCount();
    double fps = cv::getTickFrequency() / ((t1 - t0) / frames);
    std::cout << "摄像头读取 FPS: " << fps << std::endl;
    return 0;
}
