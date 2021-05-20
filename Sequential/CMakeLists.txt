cmake_minimum_required(VERSION 3.10)
project(MS LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

if(DEFINED ENV{DATAS})
    add_definitions(-Dnum_points=$ENV{DATAS})
else()
    add_definitions(-Dnum_points=1000)
endif()

## libraries
include_directories(include datas build)
set(
    INCLUDE_FILES 
    include/meanShift.hpp 
    include/utils.hpp
    include/utils.cpp
    include/geometricFunction.hpp
)
add_library(libInclude STATIC ${INCLUDE_FILES})

## C++
set(CMAKE_CXX_COMPILER)
find_package(X11 REQUIRED)
add_executable(ms_sequential main.cpp)
target_link_libraries(ms_sequential libInclude X11 -lstdc++fs -std=c++17)