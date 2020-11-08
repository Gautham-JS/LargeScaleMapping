#!/usr/bin/env python
import numpy as np
import cv2

import rospy
from math import pow, atan2, sqrt, sin, cos
from std_msgs.msg import Float64, Float64MultiArray, Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2 as pcl2

from geometry_msgs.msg import Pose

from cv_bridge import CvBridge



# window_size = 2
# min_disp = 2
# num_disp = 16
# stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
#     numDisparities = num_disp,
#     blockSize = 31,
#     P1 = 8*3*window_size**2,
#     P2 = 32*3*window_size**2,
#     disp12MaxDiff = 10,
#     uniquenessRatio = 20,
#     speckleWindowSize = 5000,
#     speckleRange = 5
# )





class image_flow:
    def __init__(self,id):
        self.id = id
        rospy.init_node('camera_driver', anonymous=True)
        rospy.Subscriber("/drone/front_camera{}/image_raw".format(id),Image,self.image_cap)

        

        pc_msg = PointCloud2()
        
        self.n_frames = 0
        self.outpath = "/home/gautham/ros_env/src/sjtu-drone/data/"

        self.bridge = CvBridge()
        self.rate = rospy.Rate(0.5)
        self.break_flg = False

    def image_cap(self,im):
        frame = self.bridge.imgmsg_to_cv2(im)
        cv2.imshow("frame{}".format(self.id),frame)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            self.break_flg = True
        elif k%256 == 32:
            cv2.imwrite("{}{}.jpg".format(self.id,self.n_frames), frame)
            print("written frame {} form camera {}".format(self.n_frames, self.id))
            self.n_frames+=1

    def run(self):
        while True:
            if self.break_flg:
                break
            self.rate.sleep()

class Stereo_Driver:
    def __init__(self):
        rospy.init_node('camera_driver', anonymous=True)
        self.id = 1
        rospy.Subscriber("/drone/front_camera1/image_raw",Image,self.image_cap1)
        self.id = 2
        rospy.Subscriber("/drone/front_camera2/image_raw",Image,self.image_cap2)
        
        self.pcpub = rospy.Publisher("/Stereo/PointCloud",PointCloud2)

        self.n_frames = 0
        self.outpath = "/home/gautham/ros_env/src/sjtu-drone/data/"

        self.bridge = CvBridge()
        self.rate = rospy.Rate(5)
        self.break_flg = False

        self.frame0 = None
        self.frame1 = None
        self.points = None

        self.disparity = None
        self.pointCloud = None

    def image_cap1(self,im):
        frame = self.bridge.imgmsg_to_cv2(im)
        self.frame0 = frame
    
    def image_cap2(self,im):
        frame = self.bridge.imgmsg_to_cv2(im)
        self.frame1 = frame

    def stereo_core(self):
        if self.frame0 is not None:
            imgL = cv2.blur(cv2.cvtColor(self.frame0, cv2.COLOR_RGB2GRAY),(5,5))
            imgR = cv2.blur(cv2.cvtColor(self.frame1, cv2.COLOR_RGB2GRAY),(5,5))
            window_size = 2
            min_disp = 2
            num_disp = 16
            stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                numDisparities = num_disp,
                blockSize = 31,
                P1 = 8*3*window_size**2,
                P2 = 32*3*window_size**2,
                disp12MaxDiff = 10,
                uniquenessRatio = 20,
                speckleWindowSize = 5000,
                speckleRange = 5
            )

            disparity = stereo.compute(imgL,imgR)
            #print(disparity[:5])
            self.disparity = disparity

            h, w = imgL.shape[:2]
            f = 0.5*w  
            Q = np.float32([[1, 0, 0, -0.5*w],
                            [0,-1, 0,  0.5*h], 
                            [0, 0, 0,     -f], 
                            [0, 0, 1,      0]])


            points = cv2.reprojectImageTo3D(disparity, Q)

            # reflect_matrix = np.identity(3)
            # reflect_matrix[0] *= -1
            # points = np.matmul(points,reflect_matrix)

            colors = cv2.cvtColor(self.frame0, cv2.COLOR_BGR2RGB)

            mask = disparity > disparity.min()
            out_colors = colors[mask]
            out_colors = out_colors.reshape(-1, 3)
            out_points = points[mask]
            points_refined = np.zeros_like(out_points)
            points_refined[:,2] = out_points[:,1]
            points_refined[:,1] = out_points[:,2]
            points_refined[:,0] = out_points[:,0]
            self.points = points_refined
        
        else:
            print("NULL Disparity")

    def run(self):
        while True:
            if self.frame0 is not None:
                cv2.imshow("Frame0",self.frame0)
                cv2.imshow("Frame1",self.frame1)
                self.stereo_core()
                cv2.imshow("disparity",self.disparity)
                h = Header()
                h.stamp = rospy.Time.now()
                h.frame_id = "map"

                scaled_points = pcl2.create_cloud_xyz32(h,self.points)
                self.pcpub.publish(scaled_points)
                print("Publishing pcl")

                k = cv2.waitKey(1)
                if k%256 == 27:
                    print("Escape hit, closing...")
                    break

                elif k%256==32:
                    print("Writing frames {}".format(self.n_frames))
                    cv2.imwrite(self.outpath+"{}_0.jpg".format(self.n_frames),self.frame0)
                    cv2.imwrite(self.outpath+"{}_1.jpg".format(self.n_frames),self.frame1)
                self.n_frames+=1
            else:
                print("NULL FRAME")
            if self.break_flg:
                break
            self.rate.sleep()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    rig1 = Stereo_Driver()
    rig1.run()




