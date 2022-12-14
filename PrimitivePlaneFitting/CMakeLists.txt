cmake_minimum_required(VERSION 3.9.0)

project(PrimitiveFitting)

set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
option (ENABLE_COVERAGE "Use gcov" OFF)

if (CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_BUILD_TYPE "debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g -pthread")
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++14 -mavx -fPIC -pthread")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -mavx -fPIC -pthread")
endif()

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(OPEN_SOURRCE_DIR ${PROJECT_SOURCE_DIR}/open_source)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})

include_directories(${CMAKE_CURRENT_LIST_DIR}/3rd/eigen)
find_package(CGAL)
include(${CGAL_USE_FILE})

add_executable(
  region_growing_3d
  PlaneDetection3D/region_growing_3d.cpp
)

target_link_libraries(
  region_growing_3d
)
