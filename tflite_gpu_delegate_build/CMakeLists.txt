cmake_minimum_required(VERSION 3.10)
project(TFLiteGpuDelegate)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)

include_directories(
    /home/rasmus/tensorflow
    /home/rasmus/tensorflow/third_party
    /home/rasmus/tensorflow/tensorflow/lite
    /home/rasmus/tensorflow/tensorflow/lite/delegates/gpu
)

set(SOURCES
    /home/rasmus/tensorflow/tensorflow/lite/delegates/gpu/api.cc
    /home/rasmus/tensorflow/tensorflow/lite/delegates/gpu/delegate.cc
    /home/rasmus/tensorflow/tensorflow/lite/delegates/gpu/delegate_options.cc
    /home/rasmus/tensorflow/tensorflow/lite/delegates/gpu/gl_delegate.cc
    /home/rasmus/tensorflow/tensorflow/lite/delegates/gpu/common/task/buffer_desc.cc
)

add_library(tflite_gpu_delegate SHARED ${SOURCES})
target_link_libraries(tflite_gpu_delegate ${OPENGL_gl_LIBRARY})
