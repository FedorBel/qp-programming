cmake_minimum_required(VERSION 3.0.0)
project(simple_qp VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(Eigen3 REQUIRED)

add_executable(simple_qp eigen.cpp)

target_link_libraries(simple_qp Eigen3::Eigen)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
