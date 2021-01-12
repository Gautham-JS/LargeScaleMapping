#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;

class StereoProcess{
    public:
        const char*lFptr; const char*rFptr;
        
        double baseline = 0.5707;
        double focal_x = 7.188560000000e+02;
        double cx = 6.071928000000e+02;
        double focal_y = 7.188560000000e+02;
        double cy = 1.852157000000e+02;

        cv::Mat K = (cv::Mat1d(3,3) << focal_x, 0, cx, 0, focal_y, cy, 0, 0, 1);

        cv::Mat lImg, rImg;
        vector<cv::Point3f> tri3dPoints;

        StereoProcess(const char* lptr, const char* rptr){
            lFptr = lptr;
            rFptr = rptr;
        }
    
        cv::Mat getImg(const char* fptr, int iter);
        void stereoTriangulate(cv::Mat im1, cv::Mat im2, vector<cv::Point3f>&out3d);
        cv::Mat stereoMatch(int iter);
        void reprojectDisparity(cv::Mat disp, vector<cv::Point3f>&reproject3dPoints);
        void visualizeCloud(vector<cv::Point3f>pts3d);
};
