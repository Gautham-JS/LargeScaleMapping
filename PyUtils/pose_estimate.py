import cv2
import numpy as np
import  cv2 as cv
import matplotlib.pyplot as plt
import math
import imutils
import time

cam = cv2.VideoCapture(0)
K = [[647.8849454418848, 0.0, 312.70216601346215],
        [0.0, 648.2741486235716, 245.95593954674428],
        [0.0, 0.0, 1.0]]
K = np.array(K)

distCoeffs = [0.035547431486979156, -0.15592121266783593, 0.0005127230470698213, -0.004324823776384423, 1.2415990279352762]
distCoeffs = np.array(distCoeffs)

focal_len = 648.2741486235716
pp = (312.70216601346215, 245.95593954674428)

im1 = cv2.imread("/home/gautham/Documents/Codes/depth_reconstruct/opencv_frame_0.png")
im2 = cv2.imread("/home/gautham/Documents/Codes/depth_reconstruct/opencv_frame_3.png")

ref_vocab = []


class PoseEstimate:
    def __init__(self,K,distCoeff):
        self.K = K
        self.distCoeff = distCoeff
        
        self.ref_frame = None
        self.frame = None
        self.ref_kpmask = None
        self.frame_kpmask = None
        self.match_frame = None
        self.Kp_ref = None
        self.descs_ref = None
        self.Kp_frame = None
        self.descs_frame = None
        self.F = None

        self.Rmat = None
        self.Tvec = None

        self.goodkps = None

        self.pts_ref = None
        self.pts_frame = None
    

        self.focal_len = 1.0
        self.pp = (0. ,0.)

        self.RotX = 0.
        self.RotY = 0.
        self.RotZ = 0.
        self.PosX = 0.
        self.PosY = 0.
        self.PosZ = 0.

        self.eulerAngles = None

        self.ref_homography = None
        self.frame_homography = None

        self.resampleFlag = False
        self.interruptFlag = False

    def featureEst(self):
        orb = cv2.ORB_create()
        self.Kp_ref, self.descs_ref = orb.detectAndCompute(self.ref_frame,None)
        self.Kp_frame, self.descs_frame = orb.detectAndCompute(self.frame,None)

    def featureMatch(self):
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(self.descs_ref, self.descs_frame, k=2)

        good = []
        pt1 = []
        pt2 = []

        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append(m)
                pt2.append(self.Kp_frame[m.trainIdx].pt)
                pt1.append(self.Kp_ref[m.queryIdx].pt)

        self.goodkps = good
        pts1 = np.float32(pt1)
        pts2 = np.float32(pt2)
        self.ref_kpmask = cv2.drawKeypoints(self.ref_frame, self.Kp_ref, self.ref_kpmask,color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        self.frame_kpmask = cv2.drawKeypoints(self.frame, self.Kp_frame, self.frame_kpmask,color=(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        #self.match_frame = cv2.drawMatchesKnn(self.ref_frame, self.Kp_ref ,self.frame, self.Kp_frame, good[:10], np.copy(self.match_frame), flags=2)

        self.F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]

        pts11 = pts1.reshape(-1,1,2)
        pts22 = pts2.reshape(-1,1,2)

        pts1_norm = cv2.undistortPoints(pts11, cameraMatrix=self.K, distCoeffs=self.distCoeff)
        pts2_norm = cv2.undistortPoints(pts22, cameraMatrix=self.K, distCoeffs=self.distCoeff)

        self.pts_ref = pts1_norm
        self.pts_frame = pts2_norm

    def vectorizePose(self):
        E,mask = cv.findEssentialMat(self.pts_ref, self.pts_frame, focal=1.0, pp=(0.,0.), method=cv2.RANSAC,prob=0.99, threshold=1.0)
        r1, r2, t = cv.decomposeEssentialMat(E)
        _,R,T,mask = cv.recoverPose(E, self.pts_ref, self.pts_frame, focal=1.0, pp=(0.,0.))

        self.Rmat = R
        self.Tvec = T

    def geometricTransform(self):
        M_r = np.hstack((self.Rmat, self.Tvec))
        projMat = np.dot(self.K, M_r)

        eulerAngles = cv2.decomposeProjectionMatrix(projMat)[-1]
        self.RotX = eulerAngles[0]
        self.Roty = eulerAngles[1]
        self.Rotz = eulerAngles[2]
        self.eulerAngles = eulerAngles

    def poseTrack(self,ref_image,image):
        self.ref_frame = ref_image
        self.frame = image


    def estimate(self):
        self.featureEst()
        self.featureMatch()
        self.vectorizePose()
        self.geometricTransform()

        return self.eulerAngles



_, ref_frame = cam.read()

xrot = []
yrot = []
zrot = []
step = []
i=0
xr = 0
yr = 0
zr = 0

pose = PoseEstimate(K,distCoeffs)
step = 0
while True:
    imbuf = imutils.rotate(im2,step)
    pose.poseTrack(im2,imbuf)
    angles = pose.estimate()

    imR = pose.ref_kpmask
    imL = pose.frame_kpmask
    #match_img = cv2.drawMatchesKnn(pose.ref_kpmask, pose.Kp_ref, pose.frame_kpmask, pose.Kp_frame, pose.goodkps,None,flags=2)

    print(np.transpose(angles))
    cv2.namedWindow("Test")
    cv2.imshow("Test", imR)

    print(len(pose.pts_frame))

    cv2.namedWindow("Main")
    cv2.imshow("Main", imL)
    step+=10
    k = cv.waitKey(0)
    if step>90:
        print("Angular Overflow")
        break
    if k%256 == 27:
        print("Escape hit, closing...")
        break

cv.destroyAllWindows()
