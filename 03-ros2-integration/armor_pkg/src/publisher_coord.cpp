#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <chrono>
#include "geometry_msgs/msg/point.hpp"

int main(int argc, char** argv){
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("coord_publisher");

    //发布 Point类型
    auto pub = node->create_publisher<geometry_msgs::msg::Point>("armor_coord", 10);
    
    rclcpp::WallRate rate(2);
    while(rclcpp::ok()){
        auto msg = geometry_msgs::msg::Point();
        msg.x = 320.0;
        msg.y = 240.0;
        msg.z = 2500.0;
        pub->publish(msg);
        RCLCPP_INFO(node->get_logger(), "发布坐标：(%.1f, %.1f, %.1f)", msg.x, msg.y,msg.z);
        rate.sleep();
        
    }
    rclcpp::shutdown();
    return 0;
}