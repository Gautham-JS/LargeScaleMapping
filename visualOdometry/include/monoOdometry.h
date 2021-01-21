/*
-->GauthWare, LSM, 01/2021
*/
#ifndef ODOM_H
#define ODOM_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>
#include <vector>
#include <algorithm>

#include "poseGraph.h"

using namespace cv;
using namespace std;

struct imageMetadata{
    Mat im1, im2, R, t;
    vector<Point2f> refPts, trkPts;
    vector<Point3f> pts3d;
};


class monoOdom{
    public:
        int iter = 0;
        int idx = 0;
        const char* lFptr;
        const char* rFptr;
        Mat im1, im2, R, t, rvec, tvec;

        double focal_x = 7.188560000000e+02;
        double cx = 6.071928000000e+02;
        double focal_y = 7.188560000000e+02;
        double cy = 1.852157000000e+02;

        Mat canvas = Mat::zeros(1000,1500, CV_8UC3);
        Mat debug1, debug2, debug3; 
        Mat K = (Mat1d(3,3) << focal_x, 0, cx, 0, focal_y, cy, 0, 0, 1);
        Mat im1prev, im2prev;
        
        imageMetadata imMeta;
        imageMetadata prevImMeta;

        globalPoseGraph poseGraph;

        monoOdom(int seq, const char* lptr, const char* rptr){
            lFptr = lptr; rFptr = rptr;
        }
        void relocalize(int start, Mat imL, Mat imR, Mat&inv_transform, vector<Point2f>&ftrPts, vector<Point3f>&pts3d);
        void PyrLKtrackFrame2Frame(Mat refimg, Mat curImg, vector<Point2f>refPts, vector<Point3f>ref3dpts,
                                            vector<Point2f>&refRetpts, vector<Point3f>&ref3dretPts);
        vector<KeyPoint> denseKeypointExtractor(Mat img, int stepSize);
        void stageForPGO(Mat Rl, Mat tl, Mat Rg, Mat tg, bool loopClose);
        Mat drawDeltas(Mat im, vector<Point2f> in1, vector<Point2f> in2);
        void monoTriangulate(Mat img1, Mat img2,vector<Point2f>&ref2dPts, vector<Point2f>&trk2dPts,vector<Point3f>&ref3dpts);
        Mat drawDeltasErr(Mat img1, vector<Point2f>inlier1, vector<Point2f>inlier2);
        void pyrLKtracking(Mat refImg, Mat curImg, vector<Point2f>&refPts, vector<Point2f>&trackPts);
        void FmatThresholding(vector<Point2f>&refPts, vector<Point2f>&trkPts);
        void relocalizeFrames(int start, Mat img1, Mat img2, Mat&invTransform, vector<Point2f>&ftrPts, vector<Point3f>pts3d);
        Mat loadImage(int iter);
        void initSequence();
        void loopSequence();
};

#endif