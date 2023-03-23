import numpy as np
import cv2


minDisparity = 16
numDisparities = 192 - minDisparity
blockSize = 5
uniquenessRatio = 1
speckleWindowSize = 3
speckleRange = 3
disp12MaxDiff = 200
P1 = 600
P2 = 2400

stereo = cv2.StereoSGBM_create(
    minDisparity = minDisparity,
    numDisparities = numDisparities,
    blockSize = blockSize,
    uniquenessRatio = uniquenessRatio,
    speckleRange = speckleRange,
    speckleWindowSize = speckleWindowSize,
    disp12MaxDiff = disp12MaxDiff,
    P1 = P1,
    P2 = P2
)

imgL = cv2.imread('2/pic1.jpg')
imgR = cv2.imread('2/pic2.jpg')

imgL = cv2.resize(imgL, (800, 600))
imgR = cv2.resize(imgR, (800, 600))

disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

cv2.imshow('Left', imgL)
cv2.imshow('Right', imgR)
cv2.imshow('Disparity',
            (disparity - minDisparity) / numDisparities)

cv2.waitKey()
