cmake_minimum_required(VERSION 3.18)

project(tensorflow_v2_cpp)

if(WIN32)
	message(INFO "This package is not tested on Windows. Feel free to report if it works!")
endif()

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# We installed Tensorflow in /opt/tensorflow - change if you installed elsewhere
set(TENSORFLOW_LIB_DIR "/opt/tensorflow/lib") 

add_executable(get_prediction get_prediction.cpp)
target_include_directories(get_prediction PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(get_prediction PRIVATE ${TENSORFLOW_LIB_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(get_prediction ${TENSORFLOW_LIB_DIR}/libtensorflow_cc.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so ${OpenCV_LIBRARIES})
