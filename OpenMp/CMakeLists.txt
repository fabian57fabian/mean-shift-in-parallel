cmake_minimum_required(VERSION 3.10)
project(MS LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

## OpenMp compiler
set(CMAKE_CXX_COMPILER "/opt/local/bin/clang++-mp-3.9")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")

## libraries
include_directories(include datas)
set(INCLUDE_FILES 
    include/meanShiftOmp.hpp 
    include/utils.hpp
    include/utils.cpp
    include/geometricFunction.hpp)

add_library(libInclude STATIC ${INCLUDE_FILES})

## CPP
set(CMAKE_CXX_COMPILER)
find_package(X11 REQUIRED)

# Static
add_executable(ms_omp_static static.cpp)
target_link_libraries(ms_omp_static libInclude X11 -lstdc++fs -std=c++17 -fopenmp)

# Dynamic
add_executable(ms_omp_dynamic dynamic.cpp)
target_link_libraries(ms_omp_dynamic libInclude X11 -lstdc++fs -std=c++17 -fopenmp)