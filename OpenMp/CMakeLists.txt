cmake_minimum_required(VERSION 3.10)
project(MS LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

## OpenMp compiler
set(CMAKE_CXX_COMPILER "/opt/local/bin/clang++-mp-3.9")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")

## libraries
include_directories(include datas)
# set(INCLUDE_FILES 
#     include/meanShift.hpp 
#     include/utils.hpp
#     include/utils.cpp
#     include/geometricFunction.hpp)
set(INCLUDE_FILES 
    include/utils.hpp
    include/utils.cpp)
add_library(libInclude STATIC ${INCLUDE_FILES})

## C++
set(CMAKE_CXX_COMPILER)
find_package(X11 REQUIRED)
add_executable(ms_openmp main.cpp)
target_link_libraries(ms_openmp libInclude X11 -lstdc++fs -std=c++17 -fopenmp)