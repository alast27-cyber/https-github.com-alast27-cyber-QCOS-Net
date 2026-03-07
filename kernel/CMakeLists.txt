cmake_minimum_required(VERSION 3.10)
project(qcos_ipde)

set(CMAKE_CXX_STANDARD 17)

# External Dependencies (Example)
# find_package(Torch REQUIRED)

add_library(qcos_ipde SHARED qcos_inference_stub.cpp)
# target_link_libraries(qcos_ipde "${TORCH_LIBRARIES}")
