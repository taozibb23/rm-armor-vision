cmake_minimum_required(VERSION 3.8)
project(armor_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(vision_node src/vision_node.cpp src/armor_algo.cpp)
ament_target_dependencies(vision_node rclcpp geometry_msgs)
target_link_libraries(vision_node ${OpenCV_LIBS})

add_executable(publisher src/publisher.cpp)
target_link_libraries(publisher ${rclcpp_LIBRARIES})
ament_target_dependencies(publisher rclcpp std_msgs)

add_executable(subscriber src/subscriber.cpp)
target_link_libraries(subscriber ${rclcpp_LIBRARIES})
ament_target_dependencies(subscriber rclcpp std_msgs)

add_executable(publisher_coord src/publisher_coord.cpp)
target_link_libraries(publisher_coord ${rclcpp_LIBRARIES})
ament_target_dependencies(publisher_coord rclcpp geometry_msgs)

add_executable(image_pub src/image_pub.cpp src/armor_algo.cpp)
target_link_libraries(image_pub ${rclcpp_LIBRARIES} ${OpenCV_LIBS})
ament_target_dependencies(image_pub rclcpp sensor_msgs cv_bridge image_transport geometry_msgs)

add_executable(armor_yolo_node src/armor_yolo_node.cpp)
set_target_properties(armor_yolo_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "/home/user/yolo_env/lib/python3.10/site-packages/onnxruntime/capi")
target_include_directories(armor_yolo_node PRIVATE
  /home/user/vision_learn/onnxruntime_include)          # ONNX Runtime 头文件
target_link_libraries(armor_yolo_node ${OpenCV_LIBS}
  /home/user/yolo_env/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2)
ament_target_dependencies(armor_yolo_node rclcpp sensor_msgs cv_bridge)

install(TARGETS publisher subscriber publisher_coord vision_node image_pub armor_yolo_node DESTINATION lib/${PROJECT_NAME})

ament_package()
