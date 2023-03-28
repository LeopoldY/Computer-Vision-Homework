import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_NUM_GOOD_MATCHES = 10

img0_oc = cv2.imread('./3/pic1.jpg')
img1_oc = cv2.imread('./3/pic2.jpg')

img0 = cv2.cvtColor(img0_oc, cv2.COLOR_RGB2GRAY)
img1 = cv2.cvtColor(img1_oc, cv2.COLOR_RGB2GRAY)
merged = np.hstack((img0_oc, img1_oc))

merged = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)

# Perform SIFT feature detection and description.
sift = cv2.ORB_create()
kp0, des0 = sift.detectAndCompute(img0, None)
kp1, des1 = sift.detectAndCompute(img1, None)

# Perform brute-force KNN matching.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
pairs_of_matches = bf.knnMatch(des0, des1, k=2)

# Sort the pairs of matches by distance.
pairs_of_matches = sorted(pairs_of_matches, key=lambda x:x[0].distance)

# Find all the good matches as per Lowe's ratio test.
good_matches = []
good_mask = []
bad_matches = []
bad_mask = []
for x in pairs_of_matches:
    if len(x) > 1 and x[0].distance < 0.8 * x[1].distance:
        good_matches.append(x)
        good_mask.append(1)
        bad_mask.append(0)
    else:
        bad_matches.append(x)
        good_mask.append(0)
        bad_mask.append(1)

# Draw the 25 best pairs of matches.
good = cv2.drawMatchesKnn(
    img0, kp0, img1, kp1, good_matches[:25], merged, 
    matchColor=(0,255,0),flags=3)

# Draw the bad 25 matches.
bad = cv2.drawMatchesKnn(
    img0, kp0, img1, kp1, bad_matches[:25], merged, 
    matchColor=(255,0,0), flags=3)

# Show the matches.
plt.imshow(merged)
plt.show()