import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_NUM_GOOD_MATCHES = 10

img0_oc = cv2.imread('./3/pic1.jpg')
img1_oc = cv2.imread('./3/pic2.jpg')

img0 = cv2.cvtColor(img0_oc, cv2.COLOR_RGB2GRAY)
img1 = cv2.cvtColor(img1_oc, cv2.COLOR_RGB2GRAY)

img0_oc = cv2.cvtColor(img0_oc, cv2.COLOR_BGR2RGB)
img1_oc = cv2.cvtColor(img1_oc, cv2.COLOR_BGR2RGB)

merged = np.hstack((img0_oc, img1_oc))

# Perform SIFT feature detection and description.
sift = cv2.SIFT_create()
kp0, des0 = sift.detectAndCompute(img0, None)
kp1, des1 = sift.detectAndCompute(img1, None)

# Define FLANN-based matching parameters.
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Perform FLANN-based matching.
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des0, des1, k=2)

# Find all the good matches as per Lowe's ratio test.
good_matches = []
bad_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
    else:
        bad_matches.append(m)


# src_pts和dst_pts分别是源图像和目标图像的匹配点坐标的numpy数组。
src_pts = np.float32(
    [kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32(
    [kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
mask_matches = mask.ravel().tolist()

# Draw the matches that passed the ratio test.
img_matches = cv2.drawMatches(
    img0_oc, kp0, img1_oc, kp1, good_matches, merged,
    matchColor=(0, 255, 0), singlePointColor=None,
    matchesMask=mask_matches, flags=3)

# src_pts和dst_pts分别是源图像和目标图像的匹配点坐标的numpy数组。
src_pts_bad = np.float32(
    [kp0[m.queryIdx].pt for m in bad_matches]).reshape(-1, 1, 2)
dst_pts_bad = np.float32(
    [kp1[m.trainIdx].pt for m in bad_matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts_bad, dst_pts_bad, cv2.RANSAC, 5.0)
mask_matches = mask.ravel().tolist()

bad_img_matches = cv2.drawMatches(
    img0_oc, kp0, img1_oc, kp1, bad_matches, merged,
    matchColor=(255, 0, 0), singlePointColor=None,
    matchesMask=mask_matches, flags=3)


# Show the homography and good matches.
plt.imshow(merged)
plt.show()