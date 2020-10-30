//LSM Gautham J.S OCT-2020
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include <vector>
#include <iostream>
#include <map>
#include <fstream>
#include <cassert>

using namespace std;
using namespace cv;

const int IMAGE_DOWNSAMPLE = 1; 
const double FOCAL_LENGTH = 1600 / IMAGE_DOWNSAMPLE; 
const int MIN_LANDMARK_SEEN = 3; 

const std::string IMAGE_DIR = "/home/gautham/Documents/Codes/Datasets/BlenderRender2/";

const std::vector<std::string> IMAGES = {
    "0001.jpg",
    "0002.jpg",
    "0003.jpg",
    "0004.jpg",
    "0005.jpg",
    "0006.jpg",
    "0007.jpg"
};

struct SFM_metadata{
    struct ImagePose{
        cv::Mat img; 
        cv::Mat desc;
        std::vector<cv::KeyPoint> kp;

        cv::Mat T;
        cv::Mat P;

        using kp_idx_t = size_t;
        using landmark_idx_t = size_t;
        using img_idx_t = size_t;

        std::map<kp_idx_t, std::map<img_idx_t, kp_idx_t>> kp_matches; 
        std::map<kp_idx_t, landmark_idx_t> kp_landmark;

        kp_idx_t& kp_match_idx(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx][img_idx]; };
        bool kp_match_exist(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx].count(img_idx) > 0; };

        landmark_idx_t& kp_3d(size_t kp_idx) { return kp_landmark[kp_idx]; }
        bool kp_3d_exist(size_t kp_idx) { return kp_landmark.count(kp_idx) > 0; }
    };

    struct Landmark{
        cv::Point3f pt;
        int seen = 0; 
    };

    std::vector<ImagePose> img_pose;
    std::vector<Landmark> landmark;
};

struct dataType {
    cv::Point3d point;
    int red;
    int green;
    int blue; 
};
typedef dataType SpacePoint;
vector<SpacePoint> pointCloud;

void toPly(){
	ofstream outfile("/home/gautham/Documents/Codes/pointcloud.ply");
	outfile << "ply\n" << "format ascii 1.0\n" << "comment VTK generated PLY File\n";
	outfile << "obj_info vtkPolyData points and polygons : vtk4.0\n" << "element vertex " << pointCloud.size() << "\n";
	outfile << "property float x\n" << "property float y\n" << "property float z\n" << "element face 0\n";
	outfile << "property list uchar int vertex_indices\n" << "end_header\n";
	for (int i = 0; i < pointCloud.size(); i++)
	{
		Point3d point = pointCloud.at(i).point;
		outfile << point.x << " ";
		outfile << point.y << " ";
		outfile << point.z << " ";
		outfile << "\n";
	}
	cout<<"PLY SAVE DONE"<<endl;
	outfile.close();
}




class SFMtoolkit{
    public:
        SFM_metadata SFM;
        void feature_proc(){
            using namespace cv;
            using namespace cv::xfeatures2d;

            Ptr<AKAZE> feature = AKAZE::create();
            //Ptr<ORB> feature = ORB::create(10000);
            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

            namedWindow("img", WINDOW_NORMAL);
            cout<<"CKPT 1 "<<endl;

            for (auto f : IMAGES) {
                SFM_metadata::ImagePose a;

                Mat img = imread(IMAGE_DIR + f);
                assert(!img.empty());

                resize(img, img, img.size()/IMAGE_DOWNSAMPLE);
                a.img = img;
                cvtColor(img, img, COLOR_BGR2GRAY);

                feature->detect(img, a.kp);
                feature->compute(img, a.kp, a.desc);

                SFM.img_pose.emplace_back(a);
            }
        }

        void FeatureMatch(){
            using namespace cv;
            using namespace xfeatures2d;
            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
            cout<<"CKPT 2 "<<endl;
            int count = 0;

            for (size_t i=0; i < SFM.img_pose.size()-1; i++) {
                auto &img_pose_i = SFM.img_pose[i];
                for (size_t j=i+1; j < SFM.img_pose.size(); j++) {
                    auto &img_pose_j = SFM.img_pose[j];
                    vector<vector<DMatch>> matches;
                    vector<Point2f> src, dst;
                    vector<uchar> mask;
                    vector<int> i_kp, j_kp;


                    matcher->knnMatch(img_pose_i.desc, img_pose_j.desc, matches, 2);

                    for (auto &m : matches) {
                        if(m[0].distance < 0.7*m[1].distance) {
                            src.push_back(img_pose_i.kp[m[0].queryIdx].pt);
                            dst.push_back(img_pose_j.kp[m[0].trainIdx].pt);

                            i_kp.push_back(m[0].queryIdx);
                            j_kp.push_back(m[0].trainIdx);
                        }
                    }


                    findFundamentalMat(src, dst, FM_RANSAC, 3.0, 0.99, mask);

                    Mat canvas = img_pose_i.img.clone();
                    Mat buffer_canvas = img_pose_j.img.clone();
                    cv::drawKeypoints(canvas, img_pose_i.kp, canvas,Scalar(255,0,0));
                    cv::drawKeypoints(buffer_canvas, img_pose_j.kp, buffer_canvas,Scalar(0,0,255));
                    canvas.push_back(buffer_canvas);

                    for (size_t k=0; k < mask.size(); k++) {
                        if (mask[k]) {
                            img_pose_i.kp_match_idx(i_kp[k], j) = j_kp[k];
                            img_pose_j.kp_match_idx(j_kp[k], i) = i_kp[k];
                            line(canvas, src[k], dst[k] + Point2f(0, img_pose_i.img.rows), Scalar(0, 255, 0), 1);
                        }
                    }

                    int good_matches = sum(mask)[0];
                    if(good_matches<10){
                        continue;
                    }
                    assert(good_matches >= 10);

                    cout << "Feature matching " << i << " " << j << " ==> " << good_matches << "/" << matches.size() << endl;

                    resize(canvas, canvas, canvas.size()/2);
                    if(count>0){
                        cv::namedWindow("img");
                        imshow("img", canvas);
                        waitKey(1000);
                    }
                    count++;
                }
                    cv::destroyAllWindows();
            }
        }

        void sfm_reconstruct(){
            using namespace cv;
            using namespace viz;

            double cx = SFM.img_pose[0].img.size().width/2;
            double cy = SFM.img_pose[0].img.size().height/2;

            Point2d pp(cx, cy);

            Mat K = Mat::eye(3, 3, CV_64F);

            K.at<double>(0,0) = FOCAL_LENGTH;
            K.at<double>(1,1) = FOCAL_LENGTH;
            K.at<double>(0,2) = cx;
            K.at<double>(1,2) = cy;

            cout << endl << "initial camera matrix K " << endl << K << endl << endl;

            SFM.img_pose[0].T = Mat::eye(4, 4, CV_64F);
            SFM.img_pose[0].P = K*Mat::eye(3, 4, CV_64F);

            for (size_t i=0; i < SFM.img_pose.size() - 1; i++) {
                auto &prev = SFM.img_pose[i];
                auto &cur = SFM.img_pose[i+1];

                vector<Point2f> src, dst;
                vector<size_t> kp_used;

                for (size_t k=0; k < prev.kp.size(); k++) {
                    if (prev.kp_match_exist(k, i+1)) {
                        size_t match_idx = prev.kp_match_idx(k, i+1);

                        src.push_back(prev.kp[k].pt);
                        dst.push_back(cur.kp[match_idx].pt);

                        kp_used.push_back(k);
                    }
                }

                Mat mask;

                Mat E = findEssentialMat(dst, src, FOCAL_LENGTH, pp, RANSAC, 0.999, 1.0, mask);
                Mat local_R, local_t;

                cout<<E<<endl;

                recoverPose(E, dst, src, local_R, local_t, FOCAL_LENGTH, pp, mask);

                Mat T = Mat::eye(4, 4, CV_64F);
                local_R.copyTo(T(Range(0, 3), Range(0, 3)));
                local_t.copyTo(T(Range(0, 3), Range(3, 4)));

                cur.T = prev.T*T;

                Mat R = cur.T(Range(0, 3), Range(0, 3));
                Mat t = cur.T(Range(0, 3), Range(3, 4));

                Mat P(3, 4, CV_64F);

                P(Range(0, 3), Range(0, 3)) = R.t();
                P(Range(0, 3), Range(3, 4)) = -R.t()*t;
                P = K*P;

                cur.P = P;

                Mat points4D;
                triangulatePoints(prev.P, cur.P, src, dst, points4D);

                if (i > 0) {
                    double scale = 0;
                    int count = 0;

                    Point3f prev_camera;

                    prev_camera.x = prev.T.at<double>(0, 3);
                    prev_camera.y = prev.T.at<double>(1, 3);
                    prev_camera.z = prev.T.at<double>(2, 3);

                    vector<Point3f> new_pts;
                    vector<Point3f> existing_pts;

                    for (size_t j=0; j < kp_used.size(); j++) {
                        size_t k = kp_used[j];
                        if (mask.at<uchar>(j) && prev.kp_match_exist(k, i+1) && prev.kp_3d_exist(k)) {
                            Point3f pt3d;

                            pt3d.x = points4D.at<float>(0, j) / points4D.at<float>(3, j);
                            pt3d.y = points4D.at<float>(1, j) / points4D.at<float>(3, j);
                            pt3d.z = points4D.at<float>(2, j) / points4D.at<float>(3, j);

                            size_t idx = prev.kp_3d(k);
                            Point3f avg_landmark = SFM.landmark[idx].pt / (SFM.landmark[idx].seen - 1);

                            new_pts.push_back(pt3d);
                            existing_pts.push_back(avg_landmark);
                        }
                    }

                    for (size_t j=0; j < new_pts.size()-1; j++) {
                        for (size_t k=j+1; k< new_pts.size(); k++) {
                            double s = norm(existing_pts[j] - existing_pts[k]) / norm(new_pts[j] - new_pts[k]);

                            scale += s;
                            count++;
                        }
                    }

                    assert(count > 0);

                    scale /= count;

                    cout << "image " << (i+1) << " ==> " << i << " scale=" << scale << " count=" << count <<  endl;

                    local_t *= scale;

                    Mat T = Mat::eye(4, 4, CV_64F);
                    local_R.copyTo(T(Range(0, 3), Range(0, 3)));
                    local_t.copyTo(T(Range(0, 3), Range(3, 4)));

                    cur.T = prev.T*T;

                    R = cur.T(Range(0, 3), Range(0, 3));
                    t = cur.T(Range(0, 3), Range(3, 4));

                    Mat P(3, 4, CV_64F);
                    P(Range(0, 3), Range(0, 3)) = R.t();
                    P(Range(0, 3), Range(3, 4)) = -R.t()*t;
                    P = K*P;

                    cur.P = P;

                    triangulatePoints(prev.P, cur.P, src, dst, points4D);
                }

                for (size_t j=0; j < kp_used.size(); j++) {
                    if (mask.at<uchar>(j)) {
                        size_t k = kp_used[j];
                        size_t match_idx = prev.kp_match_idx(k, i+1);

                        Point3f pt3d;

                        pt3d.x = points4D.at<float>(0, j) / points4D.at<float>(3, j);
                        pt3d.y = points4D.at<float>(1, j) / points4D.at<float>(3, j);
                        pt3d.z = points4D.at<float>(2, j) / points4D.at<float>(3, j);

                        if (prev.kp_3d_exist(k)) {
                            cur.kp_3d(match_idx) = prev.kp_3d(k);

                            SFM.landmark[prev.kp_3d(k)].pt += pt3d;
                            SFM.landmark[cur.kp_3d(match_idx)].seen++;
                        } else {
                            SFM_metadata::Landmark landmark;

                            landmark.pt = pt3d;
                            landmark.seen = 2;

                            SFM.landmark.push_back(landmark);

                            prev.kp_3d(k) = SFM.landmark.size() - 1;
                            cur.kp_3d(match_idx) = SFM.landmark.size() - 1;
                        }
                    }
                }
            }
            vector<Mat> pts_est;
            for (auto &l : SFM.landmark) {
                if (l.seen >= 3) {
                    SpacePoint pts;
                    l.pt /= (l.seen - 1);
                    pts.point.x = l.pt.x;
                    pts.point.y = l.pt.y;
                    pts.point.z = l.pt.z;
                    pts.red = 1; pts.blue = 1; pts.green = 1;
                    pointCloud.push_back(pts);
                }
            }
            toPly();
        }
};


int main(int argc, char **argv){
    SFMtoolkit stk;
    stk.feature_proc();
    stk.FeatureMatch();
    stk.sfm_reconstruct();
    cout<<"EXECUTED"<<endl;
	return 0;
}
