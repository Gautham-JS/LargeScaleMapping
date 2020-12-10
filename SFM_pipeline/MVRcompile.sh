#!/bin/sh

g++ -g -std=c++11 -O3 multiview_reconstruct.cpp -o mview -I/usr/include/eigen3 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_calib3d -lopencv_imgcodecs -lopencv_xfeatures2d -lgtsam -lboost_system -ltbb

