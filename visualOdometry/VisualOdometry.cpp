#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

class visualOdometry{
    public:
        int seqNo;
        double baseline = 0.54;

        string absPath;
        const char* lFptr; const char* rFptr;

        double focal_x = 7.188560000000e+02;
        double cx = 6.071928000000e+02;
        double focal_y = 7.188560000000e+02;
        double cy = 1.852157000000e+02;
        
        Mat K = (Mat1d(3,3) << focal_x, 0, cx, 0, focal_y, cy, 0, 0, 1);
        
        Mat referenceImg, currentImage;
        vector<Point3f> referencePoints3D;
        vector<Point2f> referencePoints2D;

        vector<Point2f> inlierReferencePyrLKPts;
        Mat canvas = Mat::zeros(600,600, CV_8UC3);

        Mat ret;

        visualOdometry(int Seq, const char*Lfptr, const char*Rfptr){
            lFptr = Lfptr;
            rFptr = Rfptr;
        }

        void stereoTriangulate(Mat im1, Mat im2, 
                            vector<Point3f>&ref3dPts, 
                            vector<Point2f>&ref2dPts){
            
            //Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create(1200);
            Ptr<FeatureDetector> detector = xfeatures2d::SURF::create(1500);
            //Ptr<FeatureDetector> detector = ORB::create(6000);
            if(!im1.data || !im2.data){
                cout<<"NULL IMG"<<endl;
                return;
            }

            vector<KeyPoint> kp1, kp2;
            detector->detect(im1, kp1);
            detector->detect(im2, kp2);

            Mat desc1, desc2;
            detector->compute(im1, kp1, desc1);
            detector->compute(im2, kp2, desc2);

            desc1.convertTo(desc1, CV_32F);
            desc2.convertTo(desc2, CV_32F);

            BFMatcher matcher;
            vector<vector<DMatch>> matches;
            matcher.knnMatch(desc1, desc2, matches, 2);

            vector<Point2f> pt1, pt2;
            for(int i=0; i<matches.size(); i++){
                DMatch &m = matches[i][0]; DMatch &n = matches[i][1];
                if(m.distance<0.8*n.distance){
                    pt1.emplace_back(kp1[m.queryIdx].pt);
                    pt2.emplace_back(kp2[m.trainIdx].pt);
                }
            }

            vector<Point3f> pts3d;

            Mat P1 = Mat::zeros(3,4, CV_64F);
            Mat P2 = Mat::zeros(3,4, CV_64F);
            P1.at<double>(0,0) = 1; P1.at<double>(1,1) = 1; P1.at<double>(2,2) = 1;
            P2.at<double>(0,0) = 1; P2.at<double>(1,1) = 1; P2.at<double>(2,2) = 1;
            P2.at<double>(0,3) = -baseline;

            P1 = K*P1;
            P2 = K*P2;

            Mat est3d;
            triangulatePoints(P1, P2, pt1, pt2, est3d);
            //cout<<est3d.size()<<endl;

            for(int i=0; i<est3d.cols; i++){
                Point3f localpt;
                localpt.x = est3d.at<float>(0,i) / est3d.at<float>(3,i);
                localpt.y = est3d.at<float>(1,i) / est3d.at<float>(3,i);
                localpt.z = est3d.at<float>(2,i) / est3d.at<float>(3,i);
                pts3d.emplace_back(localpt);
            }

            vector<Point2f> reprojection;
            for(int k=0; k<pts3d.size(); k++){
                Point2f projection; Point3f pt3d = pts3d[k];
                projection.x = pt3d.x; projection.y = pt3d.y;
                reprojection.push_back(projection);
            }
            //cout<<reprojection.size()<<" PTSIZE "<<pt1.size()<<endl;
            ret = drawDeltas(im1, pt1, reprojection);

            ref3dPts = pts3d;
            ref2dPts = pt1;
        }

        Mat drawDeltas(Mat im, vector<Point2f> in1, vector<Point2f> in2){
            Mat frame;
            im.copyTo(frame);

            for(int i=0; i<in1.size(); i++){
                Point2f pt1 = in1[i];
                Point2f pt2 = in2[i];
                line(frame, pt1, pt2, Scalar(0,255,0),1);
                circle(frame, pt1, 5, Scalar(0,0,255));
                circle(frame, pt2, 5, Scalar(255,0,0));
            }
            return frame;
        }

        void PyrLKtrackFrame2Frame(Mat refimg, Mat curImg, vector<Point2f>refPts, vector<Point3f>ref3dpts,
                                            vector<Point2f>&refRetpts, vector<Point3f>&ref3dretPts){
            vector<Point2f> trackPts;
            vector<uchar> Idx;
            vector<float> err;

            calcOpticalFlowPyrLK(refimg, curImg, refPts, trackPts,Idx, err);

            vector<Point2f> inlierRefPts;
            vector<Point3f> inlierRef3dPts;
            vector<Point2f> inlierTracked;
            vector<int> res;

            for(int j=0; j<refPts.size(); j++){
                if(Idx[j]==1){
                    inlierRefPts.push_back(refPts[j]);
                    ref3dretPts.push_back(ref3dpts[j]);
                    refRetpts.push_back(trackPts[j]);
                }
            }
            //refRetpts = inlierTracked;
            //ref3dretPts = inlierRef3dPts;
            inlierReferencePyrLKPts = inlierRefPts;
        }

        vector<int> removeDuplicates(vector<Point2f>&baseFeatures, vector<Point2f>&newFeatures,
                                    vector<int>&mask, int radius=10){
            vector<int> res;
            for(int i=0; i<newFeatures.size(); i++){
                Point2f&p2 = newFeatures[i];
                bool inRange=false;
                
                for(auto j:mask){
                    Point2f&p1 = baseFeatures[j];
                    if(norm(p1-p2)<radius){
                        inRange=true;
                        break;
                    }
                }

                if(!inRange){res.push_back(i);}
            }
            return res;
        }

        void repjojectionError(Mat im, vector<Point2f> pt2d, vector<Point3f>pts3d){

        }


        void relocalizeFrames(int start, Mat imL, Mat imR, Mat&inv_transform, vector<Point2f>&ftrPts, vector<Point3f>&pts3d){
            vector<Point2f> new2d;
            vector<Point3f> new3d;
            
            ftrPts.clear();
            pts3d.clear();

            stereoTriangulate(imL, imR, new3d, new2d);

            for(int i=0; i<new3d.size(); i++){
                Point3f pt = new3d[i];
                Point3f p;

                p.x = inv_transform.at<double>(0,0)*pt.x + inv_transform.at<double>(0,1)*pt.y + inv_transform.at<double>(0,2)*pt.z + inv_transform.at<double>(0,3);
                p.y = inv_transform.at<double>(1,0)*pt.x + inv_transform.at<double>(1,1)*pt.y + inv_transform.at<double>(1,2)*pt.z + inv_transform.at<double>(1,3);
                p.z = inv_transform.at<double>(2,0)*pt.x + inv_transform.at<double>(2,1)*pt.y + inv_transform.at<double>(2,2)*pt.z + inv_transform.at<double>(2,3);

                pts3d.emplace_back(p);
                ftrPts.emplace_back(new2d[i]);
            }
        }

        Mat loadImageL(int iter){
            char FileName[200];
            sprintf(FileName, lFptr, iter);

            Mat im = imread(FileName);
            if(!im.data){
                cout<<"yikes, failed to fetch frame, check the paths"<<endl;
            }
            return im;
        }
        Mat loadImageR(int iter){
            char FileName[200];
            sprintf(FileName, rFptr, iter);

            Mat im = imread(FileName);
            if(!im.data){
                cout<<"yikes, failed to fetch frame, check the paths"<<endl;
            }
            return im;
        }

        void initSequence(){
            int iter = 0;
            char FileName1[200], filename2[200];
            sprintf(FileName1, lFptr, iter);
            sprintf(filename2, rFptr, iter);

            // Mat imL = imread(FileName1);
            // Mat imR = imread(filename2);

            Mat imL = loadImageL(iter);
            Mat imR = loadImageR(iter);

            referenceImg = imL;

            vector<Point2f> features;
            vector<Point3f> pts3d;
            stereoTriangulate(imL, imR, pts3d, features);

            for(int iter=1; iter<4000; iter++){
                cout<<"PROCESSING FRAME "<<iter<<endl;
                currentImage = loadImageL(iter);

                vector<Point3f> refPts3d; vector<Point2f> refFeatures;
                PyrLKtrackFrame2Frame(referenceImg, currentImage, features, pts3d, refFeatures, refPts3d);
                //cout<<"     ref features "<<refPts3d.size()<<" refFeature size "<<refFeatures.size()<<endl;
                
                Mat distCoeffs = Mat::zeros(4,1,CV_64F);
                Mat rvec, tvec; vector<int> inliers;

                //cout<<refPts3d.size()<<endl;
                cout<<refFeatures.size()<<" "<<inlierReferencePyrLKPts.size()<<" "<<refPts3d.size()<<endl;

                solvePnPRansac(refPts3d, refFeatures, K, distCoeffs, rvec, tvec, false,100,8.0, 0.99, inliers);
                if(inliers.size()<10){
                    cout<<"Low inlier count at "<<inliers.size()<<", skipping frame "<<iter<<endl;
                    continue;
                }
                Mat R;
                Rodrigues(rvec, R);
    
                R = R.t();
                Mat t = -R*tvec;

                Mat inv_transform = Mat::zeros(3,4, CV_64F);
                R.col(0).copyTo(inv_transform.col(0));
                R.col(1).copyTo(inv_transform.col(1));
                R.col(2).copyTo(inv_transform.col(2));
                t.copyTo(inv_transform.col(3));

                Mat i1 = loadImageL(iter); Mat i2 = loadImageR(iter);

                relocalizeFrames(0, i1, i2, inv_transform, features, pts3d);
                referenceImg = currentImage;

                t.convertTo(t, CV_32F);
                Mat frame = drawDeltas(currentImage, inlierReferencePyrLKPts, refFeatures);

                Point2f center = Point2f(int(t.at<float>(0)) + 300, int(t.at<float>(2)) + 100);
                circle(canvas, center ,1, Scalar(0,0,255), 2);
                rectangle(canvas, Point2f(10, 30), Point2f(550, 50),  Scalar(0,0,0), cv::FILLED);

                imshow("frame", frame);
                imshow("trajectory", canvas);
                int k = waitKey(100);
                if (k=='q'){
                    break;
                }
            }
        }
};

int main(){
    const char* impathL = "/media/gautham/Seagate Backup Plus Drive/Datasets/dataset/sequences/00/image_0/%0.6d.png";
    const char* impathR = "/media/gautham/Seagate Backup Plus Drive/Datasets/dataset/sequences/00/image_1/%0.6d.png";

    vector<Point2f> ref2d; vector<Point3f> ref3d;

    visualOdometry VO(0, impathL, impathR);
    char FileName1[200], filename2[200];
    sprintf(FileName1, impathL, 0);
    sprintf(filename2, impathR, 0);

    Mat im1 = imread(FileName1);
    Mat im2 = imread(filename2);
    //VO.stereoTriangulate(im1, im2, ref3d, ref2d);
    //visualOdometry* VO = new visualOdometry(0, impathR, impathL);
    VO.initSequence();
}
