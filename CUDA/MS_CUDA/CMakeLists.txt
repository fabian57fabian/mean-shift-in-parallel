cmake_minimum_required(VERSION 3.10)
project(MM LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

## libraries
#include_directories()

set(TOOL_FILES
    utils.h
    utils.cpp
)

set(SRC_FILES
    main.cu
)

add_library(libTool STATIC ${TOOL_FILES})
add_library(libSrc STATIC ${SRC_FILES})


## C++
set(CMAKE_CXX_COMPILER)
#find_package(X11 REQUIRED)
# find_package(Threads REQUIRED)
# set(THREADS_PREFER_PTHREAD_FLAG ON)
#add_executable(execute_all main.cu)
#target_link_libraries(mm_sequential libTool libSrc
#    Threads::Threads X11 -lpng -lstdc++fs -std=c++17
#)


## CUDA
find_package(CUDA QUIET REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# CUDA version
add_executable(exec_cuda_sm main.cu)
target_link_libraries(exec_cuda_sm libTool libSrc -std=c++17)
set_property(TARGET exec_cuda_sm PROPERTY CXX_STANDARD 17)
set_property(TARGET exec_cuda_sm PROPERTY CXX_STANDARD_REQUIRED ON)

# OPENMP version
#add_executable(mm_sharedOpt CUDA/sharedOpt.cu)
#target_link_libraries(mm_sharedOpt libTool libSrc -lpng -lstdc++fs)