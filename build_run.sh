#!/bin/bash
cd build && cmake ../src && cmake --build . && ./multicore-hw1 && cd ..
