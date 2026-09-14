#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <chrono>

//写出发布者节点

int main(int argc, char** argv){
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("my_pub");

    //创建一个发布者对象，发布的消息类型为std_msgs::msg::String，主题名为"my_armor"，队列大小为10
    auto pub = node->create_publisher<std_msgs::msg::String>("my_armor", 10);

    //
    rclcpp::WallRate rate(2);
    while (rclcpp::ok()){
        auto msg = std_msgs::msg::String();
        msg.data = "装甲板坐标: x=500, y=300";
        pub->publish(msg);
        RCLCPP_INFO(node->get_logger(), "发布: %s", msg.data.c_str());
        rate.sleep();
    }
    rclcpp::shutdown();
    return 0;

}