import cv2
import numpy as np
import  cv2 as cv
import matplotlib.pyplot as plt
import math
import imutils

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
def odom(im1, im2):
    im1 = cv.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    kp1, descs1 = orb.detectAndCompute(im1,None)
    kp2, descs2 = orb.detectAndCompute(im2,None)


    #pts_1 = np.array([x.pt for x in kp1], dtype=np.float32)
    #pts_2 = np.array([x.pt for x in kp2], dtype=np.float32)

    matcher = cv2.BFMatcher()

    matches = matcher.knnMatch(descs1, descs2, k=2)

    good = []
    pt1 = []
    pt2 = []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pt2.append(kp2[m.trainIdx].pt)
            pt1.append(kp1[m.queryIdx].pt)

    pts1 = np.float32(pt1)
    pts2 = np.float32(pt2)
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    pts11 = pts1.reshape(-1,1,2)
    pts22 = pts2.reshape(-1,1,2)
    
    pts1_norm = cv2.undistortPoints(pts11, cameraMatrix=K, distCoeffs=distCoeffs)
    pts2_norm = cv2.undistortPoints(pts22, cameraMatrix=K, distCoeffs=distCoeffs)

    E,mask = cv.findEssentialMat(pts1_norm, pts2_norm, focal=focal_len, pp=pp, method=cv2.RANSAC,prob=0.99, threshold=1.0)
    r1, r2, t = cv.decomposeEssentialMat(E)
    _,R,T,mask = cv.recoverPose(E,pts1_norm,pts2_norm,focal=focal_len,pp=pp)

    M_r = np.hstack((R, T))
    proj = np.dot(K,M_r)

    print(R)

    angles = cv2.decomposeProjectionMatrix(proj)[-1]

    # y_rot = math.asin(R[2][0])
    # y_rot_angle = y_rot *(180/3.1415)
    # x_rot = math.acos(R[2][2]/math.cos(y_rot))
    # z_rot = math.acos(R[0][0]/math.cos(y_rot))
    # x_rot_angle = x_rot *(180/3.1415)
    # z_rot_angle = z_rot *(180/3.1415)

    print(angles[0],angles[1],angles[2])

    #print(E)

    def drawlines(img1,img2,lines,pts1,pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r,c = img1.shape
        img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
        img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv.circle(img1,tuple(pt1),5,color,-1)
            img2 = cv.circle(img2,tuple(pt2),5,color,-1)
        return img1,img2


    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(im1,im2,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(im2,im1,lines2,pts2,pts1)
    return img3, img5

step = 0
while True:
    imbuf = imutils.rotate(im1,step)
    imL,imR = odom(im1,imbuf)
    cv2.namedWindow("Test")
    cv2.imshow("Test", imL)

    cv2.namedWindow("Main")
    cv2.imshow("Main", imR)
    step+=10
    k = cv.waitKey(0)
    if step>90:
        print("Angular Overflow")
        break
    if k%256 == 27:
        print("Escape hit, closing...")
        break

