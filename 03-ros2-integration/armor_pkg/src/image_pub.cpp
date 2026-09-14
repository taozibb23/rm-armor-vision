#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cmath>
#include <algorithm>
#include <unistd.h>
#include <iostream>
#include <termios.h>
#include <cstring>

#include "armor_type.hpp"
#include "armor_algo.hpp"



// 阈值：trackbar 调参结果 不要写死，把全部数据先i塞进来再进行下一步
// ===== 阈值参数
//蓝色
int hminb = 88,  sminb = 9,  vminb = 255;
int hmaxb = 101, smaxb = 49,  vmaxb = 255;
//红色
int hminr = 0, sminr = 11,   vminr = 255;
int hmaxr = 26, smaxr = 71,  vmaxr = 255;
int hminr2 = 0,  sminr2 = 11, vminr2 = 255;
int hmaxr2 = 10, smaxr2 = 71, vmaxr2 = 255;
//装甲板类型
 
int main(int argc,char** argv){
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("image_pub_node");//建立节点
    auto image_pub = node->create_publisher<sensor_msgs::msg::Image>(
        "/video_image", 10);//建立发布器  叫做video_image  队列长度10

    cv::VideoCapture cap("/home/user/vision_learn/视觉考核参考/test.mp4");//写视频路径

    if(!cap.isOpened()){RCLCPP_ERROR(node->get_logger(),"Failed to open video file");return 1;}

    rclcpp::WallRate rate(30);//发布频率30Hz
    RCLCPP_INFO(node->get_logger(), "开始发布视频到 /video_image 话题");

    while(rclcpp::ok()){
       
    cv::Mat img;

    cap >> img; // 从摄像头读取一帧图像

    if (img.empty()) continue; // 如果图像为空，跳过本次循环

    bool usedRed = (argc >= 3 && std::string(argv[2]) == "red");
    int hmin = usedRed ? hminr : hminb;
    int smin = usedRed ? sminr : sminb;
    int vmin = usedRed ? vminr : vminb;
    int hmax = usedRed ? hmaxr : hmaxb;
    int smax = usedRed ? smaxr : smaxb;
    int vmax = usedRed ? vmaxr : vmaxb;

    ArmorDetect armordetect;
    cv::Mat imgHSV, imgmask;
    cv::Mat process_img;//缩小n变成640,480

    cv::resize(img, process_img, cv::Size(640, 360));
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7));
    

    cv::cvtColor(process_img, imgHSV, cv::COLOR_BGR2HSV);
    if(usedRed)
    {
        cv::Mat mask1, mask2;
        cv::inRange(imgHSV, cv::Scalar(hminr2, sminr, vminr), cv::Scalar(hmaxr2, smaxr, vmaxr), mask1);
        cv::inRange(imgHSV, cv::Scalar(hminr, sminr, vminr), cv::Scalar(hmaxr, smaxr, vmaxr), mask2);
        cv::bitwise_or(mask1, mask2, imgmask);
    }
    else
    {
        cv::Scalar lower(hmin, smin, vmin);   
        cv::Scalar upper(hmax, smax, vmax);
        cv::inRange(imgHSV, lower, upper, imgmask);
    }
    cv::morphologyEx(imgmask, imgmask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarcy;
    cv::Mat boxPtsMat; 
    std::vector<cv::Point> boxPtsInt;
    std::vector<cv::RotatedRect> lightBars;
    cv::findContours(imgmask, contours,hierarcy,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    
    for(int i = 0;i <(int)contours.size(); i++){
        if (cv::contourArea(contours[i]) > 30){
            cv::RotatedRect rRect = cv::minAreaRect(contours[i]);
            double h = rRect.size.height, w = rRect.size.width;
            if(h < w)std::swap(h,w);
            if(h / w < 3 )continue;//长宽比
            cv::boxPoints(rRect,boxPtsMat);
            std::vector<cv::Point2f> boxPts;
            lightBars.push_back(cv::minAreaRect(contours[i]));
            for( int k = 0; k < boxPtsMat.rows; k++){
                boxPts.push_back(cv::Point2f(boxPtsMat.at<float>(k,0),
                                             boxPtsMat.at<float>(k,1)));
            }
            boxPtsInt.clear();
            for(int k = 0; k<4;k++){
                boxPtsInt.push_back(cv::Point(boxPts[k].x, boxPts[k].y));
            }
            cv::polylines(process_img,boxPtsInt,true,cv::Scalar(0,255,0), 2);
            //if(lightBars.empty()){std::cout<<" 没有检测到灯带 "<<std::endl;}
            //std::cout<<"灯带角度"<<(rRect.angle + 90)<<"灯带中心"<<rRect.center<<std::endl;//angle加90  全部都以长边为标准
            //std::cout<<"检测到灯带： "<<lightBars.size() << " 个 "<<std::endl;
        }
     
    }
           //----------------------筛选灯带数量----------------
            //std::cout<<"正在配对灯带中....."<<std::endl;
            std::sort(lightBars.begin(), lightBars.end(),
                    [](const cv::RotatedRect& a, const cv::RotatedRect& b){
                        return a.center.x < b.center.x;//按中心点的x坐标升序排列 
                    });
            //筛选出角度垂直一些的灯带
            for(auto it = lightBars.begin(); it != lightBars.end();){
                double longdir = armordetect.normalizeDeg(it->angle + 90);//规划一
                if(std::abs(longdir - 110.0) > 55)
                    it = lightBars.erase(it);
                else
                    ++it;
            }
            cv::RotatedRect armor;//接受配对完的装甲板
            for(std::size_t i = 0; i < lightBars.size(); i++){
                for(std::size_t j = i + 1 ; j < lightBars.size(); j++){//i为第一个e灯带j为第二个灯带
                    if(armordetect.isValidPair(lightBars[i],lightBars[j],armor)){//配对成功
                        //std::cout<<"配对 成功 "<<i<<" + "<<j<<std::endl;
                        ArmorType armortype = armordetect.classifyArmor(armor);
                        switch (armortype)
                        {
                        case ArmorType::BIG:   std::cout<<"装甲板类型:BIG"  <<std::endl;break;
                        case ArmorType::SMALL: std::cout<<"装甲板类型:SMALL"<<std::endl;break;
                        default:               std::cout<<"装甲板类型:UNKNOWN"<<std::endl;break;    
                        }
                        // 用 isValidPair 返回的 armor 矩形直接画框
                        cv::Mat armorPtsMat;
                        cv::boxPoints(armor, armorPtsMat);
                        std::vector<cv::Point> armorPtsInt;
                        for(int k = 0; k < armorPtsMat.rows; k++){
                            float x = armorPtsMat.at<float>(k,0);
                            float y = armorPtsMat.at<float>(k,1);
                            armorPtsInt.push_back(cv::Point(cvRound(x), cvRound(y)));
                        }
                        cv::polylines(process_img, armorPtsInt, true, cv::Scalar(0,255,0), 2, cv::LINE_AA);
                        
                        //double dist = armordetect.getDistance(armor);
                        //std::cout<<"距离:"<<dist<<"mm"<<std::endl;

                    }//else std::cout<<"配对失败"<<std::endl;

                }
            
            }

            cv::imshow("小图像", process_img);
            cv::imshow("mask", imgmask);
            cv::waitKey(1);

        if(img.empty()){
            RCLCPP_INFO(node->get_logger(), "视频播放完成");
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);//重新播放 防止播放一次太快了
            continue;
        }

        cv_bridge::CvImage bridge;   //翻译官  
        bridge.encoding = "bgr8";   //像素是bgr
        bridge.image = process_img;       //框框和处理最后都画在小图上面 目前是小图处理
        bridge.header.stamp = node->now();
        bridge.header.frame_id = "camera";

        image_pub->publish(*bridge.toImageMsg());

        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}