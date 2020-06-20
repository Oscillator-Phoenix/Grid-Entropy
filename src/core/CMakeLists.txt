CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(core)

find_package(CUDA QUIET REQUIRED)

CUDA_ADD_LIBRARY(${PROJECT_NAME} 
                 core.h
                 cuda_error_check.h
                 core.cu)
                 
                 