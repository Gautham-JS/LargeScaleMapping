import numpy as np
import cv2
import glob
import yaml
#import pathlib

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
fx = 7
fy = 5

objp = np.zeros((fx*fy,3), np.float32)
objp[:,:2] = np.mgrid[0:fx,0:fy].T.reshape(-1,2)
n = 0000
objpoints = [] 
imgpoints = [] 

images = glob.glob(r'/home/gautham/Documents/Codes/depth_reconstruct/calib_imgs/*.png')

found = 0
for fname in images:
    print("calibrating")
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, corners = cv2.findChessboardCorners(gray, (fx,fy), None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, (fx,fy), corners2, ret)
        found += 1
        #half = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1) 
        cv2.imshow('img', img)
        #cv2.waitKey(10)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


data = {'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist()}

print("done.")
print(data)
with open("calibration_matrix.yaml", "w") as f:
    yaml.dump(data, f)