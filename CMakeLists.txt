cmake_minimum_required(VERSION 3.5)

project(DeepDeploy)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

include_directories(../MNN/include)
link_directories(../MNN/build)

add_executable(mobilenet ./mnn/mobilenet.cc)
target_link_libraries(mobilenet MNN)
target_link_libraries(mobilenet ${OpenCV_LIBS})

add_executable(yolov3 ./mnn/yolov3.cc)
target_link_libraries(yolov3 MNN)
target_link_libraries(yolov3 ${OpenCV_LIBS})