cmake_minimum_required(VERSION 2.8)

# set the project name
project(multicore-hw1)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# 添加子目录
add_subdirectory(config)
add_subdirectory(omp_entropy)
add_subdirectory(core)


# 指定生成目标
add_executable(${PROJECT_NAME} main.cpp)


# 包含头文件
target_include_directories(${PROJECT_NAME} PUBLIC config)
target_include_directories(${PROJECT_NAME} PUBLIC omp_entropy)
target_include_directories(${PROJECT_NAME} PUBLIC core)

# 添加链接库
target_link_libraries(${PROJECT_NAME} config)
target_link_libraries(${PROJECT_NAME} omp_entropy)
target_link_libraries(${PROJECT_NAME} core)


message("project name: " ${PROJECT_NAME})
message("source path: " ${PROJECT_SOURCE_DIR})
message("build path: " ${PROJECT_BINARY_DIR})
message("C++ standard: C++" ${CMAKE_CXX_STANDARD})




