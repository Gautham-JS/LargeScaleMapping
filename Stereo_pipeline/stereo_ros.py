import numpy as np
import cv2
from matplotlib import pyplot as plt
import open3d as o3d




def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    out_colors = colors.copy()
    verts = verts.reshape(-1, 3)
    verts = np.hstack([verts, out_colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def Pose_Est(im1,im2,k1,k2):
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
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    pts11 = pts1.reshape(-1,1,2)
    pts22 = pts2.reshape(-1,1,2)
    
    pts1_norm = cv2.undistortPoints(pts11, cameraMatrix=k1, distCoeffs=None)
    pts2_norm = cv2.undistortPoints(pts22, cameraMatrix=k2, distCoeffs=None)

    E,mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=k1[0,0], pp=(k1[0,2],k1[1,2]), method=cv2.RANSAC,prob=0.99, threshold=1.0)
    r1, r2, t = cv2.decomposeEssentialMat(E)
    _,R,T,mask = cv2.recoverPose(E,pts1_norm,pts2_norm,focal=k1[0,0],pp=(k1[0,2],k1[1,2]))
    return R,T


calib_mat1 = [[3997.684,0,1176.728],
              [0,3997.684,1011.728],
              [0,0,1]]
calib_mat2 = [[3997.684,0,1307.839],
              [0,3997.684,1011.728],
              [0,0,1]]

K1 = np.array(calib_mat1)
K2 = np.array(calib_mat2)

path = "/home/gautham/ros_env/src/sjtu-drone/data/"
path2 = "/home/gautham/ros_env/src/sjtu-drone/data/"

imgLc = cv2.imread(path + '5_0.jpg')
imgRc = cv2.imread(path + '5_1.jpg')

imgL = cv2.blur(cv2.cvtColor(imgLc, cv2.COLOR_RGB2GRAY),(5,5))
imgR = cv2.blur(cv2.cvtColor(imgRc, cv2.COLOR_RGB2GRAY),(5,5))

#imgL = cv2.resize(imgL, (0,0), fx=0.3, fy=0.7) 
#imgR = cv2.resize(imgR, (0,0), fx=0.3, fy=0.7) 

#imgL = cv2.undistort(imgL,K1,distCoeffs=None)
#imgR = cv2.undistort(imgR,K2,distCoeffs=None)

#stereo = cv2.StereoBM_create(numDisparities=288, blockSize=27)
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


#cv2.filterSpeckles()
# stereo = cv2.StereoMatcher()

disparity = stereo.compute(imgL,imgR)

# cv2.imshow("disp",disparity)
# cv2.waitKey(1)


Tmat = np.array([0.2, 0., 0.])

#R, t = Pose_Est(imgR,imgL,K2,K1)
#Tmat = t

print(np.linalg.norm(Tmat))
rev_proj_matrix = np.zeros((4,4))


cv2.stereoRectify(cameraMatrix1 = K1,cameraMatrix2 = K2, \
                  distCoeffs1 = 0, distCoeffs2 = 0, \
                  imageSize = imgL.shape[:2], \
                  R = np.identity(3), T = Tmat, \
                  R1 = None, R2 = None, \
                  P1 =  None, P2 =  None, Q = rev_proj_matrix)

h, w = imgL.shape[:2]
f = 0.5*w  
Q = np.float32([[1, 0, 0, -0.5*w],
                [0,-1, 0,  0.5*h], 
                [0, 0, 0,     -f], 
                [0, 0, 1,      0]])


points = cv2.reprojectImageTo3D(disparity, Q)
reflect_matrix = np.identity(3)
reflect_matrix[0] *= -1
points = np.matmul(points,reflect_matrix)

colors = cv2.cvtColor(imgLc, cv2.COLOR_BGR2RGB)

mask = disparity > disparity.min()
out_colors = colors[mask]
out_colors = out_colors.reshape(-1, 3)
out_points = points[mask]
# idx = np.fabs(out_points[:,0]) < 0.3
# out_points = out_points[idx]

#write_ply(path + 'out.ply', out_points, out_colors)

print(out_points)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(out_points))
o3d.visualization.draw_geometries([pcd])