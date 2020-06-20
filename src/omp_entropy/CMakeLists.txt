CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(omp_entropy)

ADD_LIBRARY(${PROJECT_NAME} 
            omp_entropy.h 
            omp_entropy.cpp)
                 

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()