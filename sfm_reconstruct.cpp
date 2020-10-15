#define CERES_FOUND true
#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::sfm;


int main(int argc, char *argv[]){
	String im1 = "/home/gautham/Documents/Codes/Datasets/templeRing/templeR0001.png";
	String im2 = "/home/gautham/Documents/Codes/Datasets/templeRing/templeR0002.png";
	String im3 = "/home/gautham/Documents/Codes/Datasets/templeRing/templeR0005.png";
	String im4 = "/home/gautham/Documents/Codes/Datasets/templeRing/templeR0010.png";
	String im5 = "/home/gautham/Documents/Codes/Datasets/templeRing/templeR0015.png";
	String im6 = "/home/gautham/Documents/Codes/Datasets/templeRing/templeR0020.png";

	vector<String> images_paths;
	vector<cv::String> impath;
    cv::glob("/home/gautham/Documents/Codes/Datasets/templeRing/*.png", impath, false);

	images_paths.push_back(im1);
	images_paths.push_back(im2);
	images_paths.push_back(im3);
	images_paths.push_back(im4);
	images_paths.push_back(im5);
	images_paths.push_back(im6);
	int x = 0;

	cout<<"Images count : "<<impath.size()<<endl;


	cout << "\nLOADED FIRST IMG" << endl;

	float f = atof(argv[2]),
		  cx = atof(argv[3]), cy = atof(argv[4]);
	Matx33d K = Matx33d(f, 0, cx,
						0, f, cy,
						0, 0, 1);
	bool is_projective = true;
	vector<Mat> Rs_est, ts_est, points3d_estimated;
	reconstruct(images_paths, Rs_est, ts_est, K, points3d_estimated, is_projective);

	cout << "\n----------------------------\n"
		 << endl;
	cout << "Reconstruction: " << endl;
	cout << "============================" << endl;
	cout << "Estimated 3D points: " << points3d_estimated.size() << endl;
	cout << "Estimated cameras: " << Rs_est.size() << endl;
	cout << "Refined intrinsics: " << endl
		 << K << endl
		 << endl;
	cout << "3D Visualization: " << endl;
	cout << "============================" << endl;
	viz::Viz3d window("Coordinate Frame");
	window.setWindowSize(Size(500, 500));
	window.setWindowPosition(Point(150, 150));
	window.setBackgroundColor();

	cout << "Recovering points  ... ";

	vector<Vec3f> point_cloud_est;
	for (int i = 0; i < points3d_estimated.size(); ++i)
		point_cloud_est.push_back(Vec3f(points3d_estimated[i]));
	cout << "[DONE]" << endl;
	cout << "Recovering cameras ... ";
	vector<Affine3d> path;
	for (size_t i = 0; i < Rs_est.size(); ++i)
		path.push_back(Affine3d(Rs_est[i], ts_est[i]));
	cout << "[DONE]" << endl;
	if (point_cloud_est.size() > 0)
	{
		cout << "Rendering points   ... ";
		viz::WCloud cloud_widget(point_cloud_est, viz::Color::green());
		window.showWidget("point_cloud", cloud_widget);
		cout << "[DONE]" << endl;
	}
	else
	{
		cout << "Cannot render points: Empty pointcloud" << endl;
	}
	if (path.size() > 0)
	{
		cout << "Rendering Cameras  ... ";
		window.showWidget("cameras_frames_and_lines", viz::WTrajectory(path, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
		window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(path, K, 0.1, viz::Color::yellow()));
		window.setViewerPose(path[0]);
		cout << "[DONE]" << endl;
	}
	else
	{
		cout << "Cannot render the cameras: Empty path" << endl;
	}
	cout << endl
		 << "Press 'q' to close each windows ... " << endl;
	window.spin();
	return 0;
}
