cmake_minimum_required(VERSION 3.10)
project(MS LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

## libraries
include_directories(include)
set(INCLUDE_FILES include/meanShift.hpp 
    include/meanShift.cpp )
add_library(libInclude ${INCLUDE_FILES})

## C++
set(CMAKE_CXX_COMPILER)
add_executable(ms_sequential main.cpp)
target_link_libraries(ms_sequential libInclude)