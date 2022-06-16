import cv2
import numpy as np
from pathlib import Path

def calculate_homography(point1, point2):

    # calculate homography matrix from 4 points
    A = np.array([
        [point1[0][0][0], point1[0][0][1], 1, 0, 0, 0, -point2[0][0][0] *
         point1[0][0][0], -point2[0][0][0] * point1[0][0][1], -point2[0][0][0]],
        [0, 0, 0, point1[0][0][0], point1[0][0][1], 1, -point2[0][0][1] *
         point1[0][0][0], -point2[0][0][1] * point1[0][0][1], -point2[0][0][1]],

        [point1[1][0][0], point1[1][0][1], 1, 0, 0, 0, -point2[1][0][0] *
         point1[1][0][0], -point2[1][0][0] * point1[1][0][1], -point2[1][0][0]],
        [0, 0, 0, point1[1][0][0], point1[1][0][1], 1, -point2[1][0][1] *
         point1[1][0][0], -point2[1][0][1] * point1[1][0][1], -point2[1][0][1]],

        [point1[2][0][0], point1[2][0][1], 1, 0, 0, 0, -point2[2][0][0] *
         point1[2][0][0], -point2[2][0][0] * point1[2][0][1], -point2[2][0][0]],
        [0, 0, 0, point1[2][0][0], point1[2][0][1], 1, -point2[2][0][1] *
         point1[2][0][0], -point2[2][0][1] * point1[2][0][1], -point2[2][0][1]],

        [point1[3][0][0], point1[3][0][1], 1, 0, 0, 0, -point2[3][0][0] *
         point1[3][0][0], -point2[3][0][0] * point1[3][0][1], -point2[3][0][0]],
        [0, 0, 0, point1[3][0][0], point1[3][0][1], 1, -point2[3][0][1] *
         point1[3][0][0], -point2[3][0][1] * point1[3][0][1], -point2[3][0][1]]
    ])

    u, s, vh = np.linalg.svd(A)
    homography = (vh[-1, :] / vh[-1, -1]).reshape(3, 3)
    return homography


def calculateHomographyMatrixByRansac(src_pts, dst_pts, threshold=5, maxIters=1000):
    # ransac algorithm for finding best homography matrix

    src_pts = np.dstack((src_pts, np.ones((src_pts.shape[0], src_pts.shape[1]))))
    dst_pts = np.dstack((dst_pts, np.ones((dst_pts.shape[0], dst_pts.shape[1]))))
    best_count_matches = 0

    for iteration in range(maxIters):
        # get four random points
        random_indices = np.random.randint(0, len(src_pts) - 1, 4)
        random_kp_src, random_kp_dst = src_pts[random_indices], dst_pts[random_indices]

        # calculate a homography
        homography = calculate_homography(random_kp_src, random_kp_dst)
        count_matches = 0

        for i in range(len(src_pts)):

            unnorm = np.dot(homography, src_pts[i][0])
            if unnorm[-1] != 0:
                normalized = (unnorm / unnorm[-1])
            else:
                normalized = unnorm
            # calculate norm
            norm = np.linalg.norm(normalized - dst_pts[i][0])
            if norm < threshold:
                count_matches += 1

        if count_matches >= best_count_matches:
            # update best homography if better match found
            best_count_matches = count_matches
            homography_best = homography
    return homography_best


def findHomography(kp1, kp2, good):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography = calculateHomographyMatrixByRansac(src_pts, dst_pts, threshold=5,
                        maxIters=1000)
    return homography


def calculate_match_points(img1, img2, homography):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    # edge points from image 1

    edge_points_1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    # edge points from image 2
    edge_points_2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # computes transformation using homography matrix
    new_points = []
    for point in edge_points_2:
        p = np.dot(homography, np.array([point[0][0], point[0][1], 1]))
        p = p / p[-1]
        new_points.append(p[:-1])
    new_points = np.array(new_points, dtype=np.float32).reshape(edge_points_2.shape)

    # match points
    matches = np.vstack((edge_points_1, new_points))

    # find new edges coordinate and round
    x_min, y_min = np.int32(matches.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(matches.max(axis=0).ravel() + 0.5)

    return x_min, y_min, x_max, y_max


def create_panorama(img1, img2, homography):
    # stitch 2 images using homography matrix
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    x_min, y_min, x_max, y_max = calculate_match_points(img1, img2, homography)

    new_height = y_max - y_min
    new_width = x_max - x_min

    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    homography =  np.dot(H_translation, homography)
    homography = np.linalg.inv(homography)

    coordinates = np.zeros((new_height, new_width, 2))

    for i in range(new_height):
        for j in range(new_width):
            unnorm = np.dot(homography, np.array([j, i, 1]))
            coordinates[i, j, :] = (unnorm / unnorm[-1])[:-1]

    # new coordinates after homography
    coordinates = coordinates.reshape(-1, 2)

    # join colors
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    # for each pixel is x0, x1, y0, y1
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1

    x0 = np.clip(x0, 0, img2.shape[1] - 1)
    x1 = np.clip(x1, 0, img2.shape[1] - 1)
    y0 = np.clip(y0, 0, img2.shape[0] - 1)
    y1 = np.clip(y1, 0, img2.shape[0] - 1)

    # weighted sum of 4 nearest neighbours
    panorama = (img2[y0, x0].T * (x1 - x) * (y1 - y)).T \
                   + (img2[y1, x0].T * (x1 - x) * (y - y0) ).T \
                   + (img2[y0, x1].T * (x - x0) * (y1 - y)).T \
                   + ( img2[y1, x1].T * (x - x0) * (y - y0)).T

    panorama = panorama.reshape(new_height, new_width, 3)
    panorama = panorama.astype(np.uint8)

    panorama[-y_min:h1 - y_min, -x_min:w1 - x_min] = img1

    # crops edges of an image
    y_nonzero, x_nonzero, _ = np.nonzero(panorama)

    panorama = panorama[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

    return panorama


def extractFeatureWithSIFT(img):

    # extract features of image
    extractor = cv2.SIFT_create()
    kp, des = extractor.detectAndCompute(img, None)
    img3 = cv2.drawKeypoints(img, kp, img)

    cv2.imshow('sift', img3)
    cv2.waitKey(0)
    return kp, des


def extractFeatureWithORB(img):

    # extract features of image
    extractor = cv2.ORB_create()
    kp, des = extractor.detectAndCompute(img, None)

    img3 = cv2.drawKeypoints(img, kp, img)
    cv2.imshow('sift', img3)
    cv2.waitKey(0)
    return kp, des


def matchFeatures(feature1, feature2, kp1, kp2, img1, img2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(feature1, feature2, k=2)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[:10], None)

    cv2.imshow('w ', img3)
    cv2.waitKey(0)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    return good


if __name__ == "__main__":
    dataset_dir = Path('cvc01passadis-cyl-pano01')
    imgs = [img for img in dataset_dir.glob("./*.png")]
    imgs = [cv2.imread(str(img)) for img in imgs]
    img_list = []
    base_image = imgs[0]

    for i in range(0, len(imgs)):
        src_img = base_image
        dst_img = imgs[i]

        kp1, des1 = extractFeatureWithSIFT(src_img)
        kp2, des2 = extractFeatureWithSIFT(dst_img)

        good = matchFeatures(des1, des2, kp1, kp2, src_img, dst_img)

        if len(good) > 1:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            homography = findHomography(kp1, kp2, good)
            if homography is not None:
                print(i , ' is homography ')
                panorama = create_panorama(dst_img, src_img, homography)
                base_image = panorama
                cv2.imshow('output_image', base_image)
                cv2.waitKey(0)
            else:
                base_image = dst_img
                cv2.imshow('output_image', base_image)
                cv2.waitKey(0)
        else:
            base_image = dst_img
            cv2.imshow('output_image', base_image)
            cv2.waitKey(0)