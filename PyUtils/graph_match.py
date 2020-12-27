import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import tqdm
import colorama as clr
import networkx as nx

from matplotlib import style
style.use("ggplot")


impath = "/home/gautham/Documents/Datasets/SfM_quality_evaluation/Benchmarking_Camera_Calibration_2008/castle-P30/images/"
#impath = "/home/gautham/Documents/Datasets/Home/"
#impath = "/home/gautham/Documents/Datasets/Navona/"

class ImageKit:
    def __init__(self):
        self.ID = -1
        self.Img = None
        self.Kp = None
        self.desc = None
        
        self.R = None
        self.t = None
        self.P = None
        self.matchlis = dict()

class InterFrame:
    def __init__(self, f_len, pp):
        self.match = None
        self.inlier1 = None
        self.inlier2 = None
        
        self.F = None

        self.focal = f_len
        self.pp = pp
        self.K1 = np.array([f_len, 0, pp[0], 0, f_len, pp[1], 0, 0, 1]).reshape(3,3)
        self.K2 = np.array([f_len, 0, pp[0], 0, f_len, pp[1], 0, 0, 1]).reshape(3,3)

        self.R = None
        self.t = None
        self.P = None

        self.mode = -1
    
    def FmatEstimate(self,pts1,pts2):
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        self.F, mask = cv2.findFundamentalMat(pts1,pts2,method=cv2.RANSAC,ransacReprojThreshold=0.1,confidence=0.99)
        i1 = []
        i2 = []
        
        k1_inv = np.linalg.inv(self.K1)
        k2_inv = np.linalg.inv(self.K2)

        self.inlier1 = pts1[mask.ravel()==1]
        self.inlier2 = pts2[mask.ravel()==1]

        for i in range(len(mask)):
            if mask[i]:
                i1.append(k1_inv.dot([pts1[i][0], pts1[i][1], 1]))
                i2.append(k2_inv.dot([pts2[i][0], pts2[i][1], 1]))

        E = self._ERtEstimate()
        R1, R2, t = cv2.decomposeEssentialMat(E)
        
        if not (self.in_front_of_both_cameras(i1, i2, R1, t)):
            self.R = R1
            self.t = t
            self.mode = 1

        elif not (self.in_front_of_both_cameras(i1, i2, R1, -1*t)):
            self.R = R1
            self.t = -1*t
            self.mode = 2

        elif not (self.in_front_of_both_cameras(i1, i2, R2, t)):
            self.R = R2
            self.t = t
            self.mode = 3

        elif not (self.in_front_of_both_cameras(i1, i2, R2, -1*t)):
            self.R = R2
            self.t = -1*t
            self.mode = 4

        else:
            pass
    
    def _ERtEstimate(self):
        E, mask = cv2.findEssentialMat(self.inlier1,self.inlier2,focal=self.focal, pp = self.pp)
        R1,R2,t = cv2.decomposeEssentialMat(E)
        self.R = cv2.Rodrigues(R2)[0]
        self.t = t
        return E
    
    def in_front_of_both_cameras(self,first_points, second_points, rot, trans):
        rot_inv = rot
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], second)
            first_3d_point = np.array([first[0] * first_z, first[1] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False

        return True


class SFMnode:
    def __init__(self):
        self.images = []
        self.matches = dict()
        self.g = nx.DiGraph()
    
    def RtTransform(self):
        Rp = np.ones((3,3))
        tp = np.ones((3,1))
        count = 0
        for i in self.images:
            if(count==0):
                count+=1
                continue
            
            for j,ifr in i.matchlis.items():
                print(j.R, i.R)
                Rp = (j.R).dot(i.R)
                tp = i.t + (Rp.dot(j.t))
                print(tp)
            print("\n\n")


    def graphLogging(self,metadata=False):
        print("\n--------GRAPH LINK SUMMARY--------")
        for i in self.images:
            links = []
            meta = []
            skp = i.Kp
            sdes = i.desc
            for j,ifr in i.matchlis.items():
                links.append(j.ID)
                self.g.add_edge(i.ID,j.ID)
            print("\nNode {} linked to : {}".format(i.ID,links))
    
    def plotLinks(self):
        nx.draw_kamada_kawai(self.g,with_labels=True)
        plt.show()
    

def rescale(im1,factor):
    scale_percent = factor
    width = int(im1.shape[1] * scale_percent / 100)
    height = int(im1.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized = cv2.resize(im1, dim, interpolation = cv2.INTER_AREA)
    return resized






counter = 0
sfm = SFMnode()

print("--------CONSTRUCTING GRAPH--------")

prevR = np.ones((3,3))
prevt = np.ones((3,1))

n_frame = 0


for f in glob.glob(impath+"*.jpg"):
    im = cv2.imread(f)

    SCALE_FACTOR = 30
    #FOCAL_LENGTH = 2780 * (SCALE_FACTOR/100)
    FOCAL_LENGTH = 3600 * (SCALE_FACTOR/100)
    #FOCAL_LENGTH = 1190 * (SCALE_FACTOR/100)
    PP = (im.shape[1]*(SCALE_FACTOR/100)//2, im.shape[0]*(SCALE_FACTOR/100)//2)
    MATCH_THRESH = 12

    im = rescale(im,SCALE_FACTOR)

    ikt = ImageKit()
    ikt.Img = im
    ikt.ID = counter
    
    #detector = cv2.xfeatures2d.SIFT_create(1000)
    detector = cv2.ORB_create(1000)
    #detector = cv2.AKAZE_create()
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)

    # flann = cv2.FlannBasedMatcher(index_params,search_params)
    bf = cv2.BFMatcher()

    kps, desc = detector.detectAndCompute(im,None)
    ikt.Kp = kps
    ikt.desc = desc
    #print("n_features in node {} : {}".format(ikt.ID,len(kps)))
    if(counter==0):
        sfm.images.append(ikt)
        n_frame+=1
        counter+=1
        continue
    counter+=1
    translation = []
    for ims in sfm.images:
        ifr = InterFrame(FOCAL_LENGTH, PP)
        matches = bf.knnMatch(ikt.desc ,ims.desc, k=2)
        good = []
        pts1 = []
        pts2 = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append(m)
                pts2.append(ims.Kp[m.trainIdx].pt)
                pts1.append(ikt.Kp[m.queryIdx].pt)
        if(len(good)<MATCH_THRESH):
            print(f"{clr.Fore.RED}unlinking bad nodes {ikt.ID} and {ims.ID} with low matches : {len(good)}{clr.Style.RESET_ALL}")
            continue
        ifr.match = good
        ifr.FmatEstimate(np.array(pts1), np.array(pts2))
        if(len(ifr.inlier1)<MATCH_THRESH or len(ifr.inlier2)<MATCH_THRESH):
            #print(f"{clr.Fore.RED}unlinking bad nodes {ikt.ID} and {ims.ID} with low Inlier count : {len(ifr.inlier1)}, {len(ifr.inlier2)}{clr.Style.RESET_ALL}")
            continue
        if(ifr.mode==-1):
            print("--X unlinking bad nodes {} and {} with uncompatible modes : {}".format(ikt.ID, ims.ID, len(good)))
            continue

    
        #print(ifr.R,ifr.t)
        print(f"{clr.Fore.GREEN}--> Node {ikt.ID} connected to {ims.ID} in mode : {ifr.mode}{clr.Style.RESET_ALL}")
        if(n_frame==1):
            prevR = ifr.R
            prevt = ifr.t
            n_frame+=1
        else:
            prevt = prevt + (ifr.R.dot(ifr.t))
            prevR = ifr.R.dot(prevR)
        
        """
        --> matched ims.image , ikt.image stored edge data in ifr   [x]
        --> extract x,y from matches    [x]
        --> compute F/E     [x]
        --> recoverpose
        --> push to ifr
        """

        ims.matchlis.update({ikt : ifr})
        ikt.matchlis.update({ims : ifr})
    n_frame=1 
    sfm.images.append(ikt)


sfm.graphLogging()
sfm.plotLinks()
#sfm.RtTransform()

for i in sfm.images:
    skp = i.Kp
    sdes = i.desc
    for j,ifr in i.matchlis.items():
        out = cv2.drawMatches(i.Img, skp, j.Img, j.Kp, ifr.match , None)
        cv2.imshow("mats",out)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


    










