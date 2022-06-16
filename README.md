# Image-Stitching-with-Keypoint-Descriptors
 
In this assignment, I merged sub-images by using keypoint description methods SIFT, and ORB and obtain a final panorama image that including all scenes in the sub-images.

![image](https://user-images.githubusercontent.com/44320909/174104324-00071a09-ec07-4c5f-9401-dceb22c80e37.png)

To run code, enter a folder's path to dataset_dir value in line 207. Example:     dataset_dir = Path('cvc01passadis-cyl-pano01')

The code use SIFT feature extraction default. To change that, in line 217 and 218, change called functions like below.

	kp1, des1 = extractFeatureWithORB(src_img)
        kp2, des2 = extractFeatureWithORB(dst_img)

All functions are calling from main step by step. 

Functions:


def extractFeatureWithSIFT(img): Extract features bu SIFT

def extractFeatureWithORB(img): Extract features bu ORB

def matchFeatures(feature1, feature2, kp1, kp2, img1, img2): do feature matching

def create_panorama(img1, img2, homography):

def calculate_match_points(img1, img2, homography):

def findHomography(kp1, kp2, good):

def calculateHomographyMatrixByRansac(src_pts, dst_pts, threshold=5, maxIters=1000):

def calculate_homography(point1, point2):

