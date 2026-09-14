#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>


int nmsTopK(const std::vector<cv::Rect>& dets){
   std::sort(dets.begin(), dets.end(),[](const Det& a, cosnt Det& b){return a.conf > b.conf;});
   std::vector<int> kept;
   for (size_t i = 0; i < dets.size(); i++){
    int sameCluster = 0;
    for(int idx : kept){
        if (iou(dets[i], dets[idx]) > KiouThr){
            sameCluster++;
        }
    }
    if(sameCluster < keepTop) kept.push_back((int)i);
   }
   return kept.size();
} 