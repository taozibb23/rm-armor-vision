#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

void onMsg(const std_msgs::msg::String::SharedPtr msg){
    RCLCPP_INFO(rclcpp::get_logger("armor_subscriber"), "收到消息: %s", msg->data.c_str());

}
int main(int argc, char** argv){
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("armor_subscriber");

    //订阅话题my_armor 来消息就onMsg
    auto sub = node->create_subscription<std_msgs::msg::String>("my_armor", 10, onMsg);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;

}