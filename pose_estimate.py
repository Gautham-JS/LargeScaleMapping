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

focal_len = 648.2741486235716
pp = (312.70216601346215, 245.95593954674428)

im1 = cv2.imread("/home/gautham/Documents/Codes/depth_reconstruct/opencv_frame_0.png")
im2 = cv2.imread("/home/gautham/Documents/Codes/depth_reconstruct/opencv_frame_3.png")

ref_vocab = []

def odom(im1, im2):
    try:
        relocalizeFlag=False
        im1 = cv.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(200)

        kp1, descs1 = orb.detectAndCompute(im1,None)
        kp2, descs2 = orb.detectAndCompute(im2,None)

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

        pts1 = np.int32(pt1)
        pts2 = np.int32(pt2)
        F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]

        mask = np.zeros_like(im2)
        for i in pts2:
            im2 = cv.circle(im2, tuple(i),10, (0,255,0))

        outimg = cv.add(im2, mask)

        E,mask = cv.findEssentialMat(pts1, pts2, focal=focal_len, pp=pp, method=cv2.RANSAC,prob=0.99, threshold=1.0)
        #r1, r2, t = cv.decomposeEssentialMat(E)
        _,R,T,mask = cv.recoverPose(E,pts1,pts2,focal=focal_len,pp=pp)

        geom_transf = np.hstack((R, T))
        proj = np.dot(K,geom_transf)
        
        M_r = np.hstack((R, T))

        angles = cv.decomposeProjectionMatrix(proj)[-1]
        
        print(angles)

        #point_3d = point_4d[:3, :].T



        y_rot = math.asin(R[2][0])
        y_rot_angle = y_rot *(180/3.1415)
        x_rot = math.acos(R[2][2]/math.cos(y_rot))
        z_rot = math.acos(R[0][0]/math.cos(y_rot))
        x_rot_angle = x_rot *(180/3.1415)
        z_rot_angle = z_rot *(180/3.1415)

        if len(pts2)<10:
            relocalizeFlag=True

        return x_rot_angle, y_rot_angle, z_rot_angle, im2, relocalizeFlag


    except AttributeError as Atib:
        print("\n\n\n----------ATR Interrupted----------\n\n\n")
        return None, None, None, np.zeros_like(im2), False
    except ValueError as Val:
        print("\n\n\n----------VAL Interrupted----------\n\n\n")
        return None, None, None, np.zeros_like(im2), False

_, ref_frame = cam.read()

xrot = []
yrot = []
zrot = []
step = []
i=0
xr = 0
yr = 0
zr = 0
while True:
    _,frame = cam.read()
    x,y,z,outimg, relocalize = odom(ref_frame,frame)

    if relocalize==True:
        print("RELOCALIZING")
        ref_frame=frame
        xr=x
        yr=y
        zr=z
    print(type(x))
    if x is not None:
        print(x+xr,y+yr,z+zr)
    xrot.append(x)
    yrot.append(y)
    zrot.append(z)
    step.append(i)
    i+=1
    cv.imshow("debug_img",outimg)
    k = cv2.waitKey(1)
    if k%256 == 27:
        print("Escape hit, closing...")
        break

cv.destroyAllWindows()
plt.plot(step,xrot)
plt.plot(step,yrot)
plt.plot(step,zrot)
plt.show()