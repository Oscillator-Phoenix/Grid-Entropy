CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(config)

ADD_LIBRARY(${PROJECT_NAME} 
             config.h
             utils.h
             samples.h
             samples.cpp)
                 