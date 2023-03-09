import numpy as np
import cv2
from scipy import ndimage

# 将路径修改为你的路径
lena = cv2.imread("/Users/leopold/Documents/ComputerVisionHomework/1/Homework.jpeg", cv2.IMREAD_GRAYSCALE)
blured = cv2.GaussianBlur(lena, (51, 51), 0)

# sobel
kernelSobel_X = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
kernelSobel_Y = np.array([[-1,-2,-1],
                          [ 0, 0, 0],
                          [ 1, 2, 1]])
lenaSobel_X = ndimage.convolve(blured, kernelSobel_X)
lenaSobel_Y = ndimage.convolve(blured, kernelSobel_Y)

# Laplacian
kernelLaplacian = np.array([[ 0, 1, 0],
                            [ 1,-4, 1],
                            [ 0, 1, 0]])
lenaLaplacian = ndimage.convolve(blured, kernelLaplacian)

# scharr
kernalScharr_X = np.array([[ -3, 0,  3],
                           [-10, 0, 10],
                           [ -3, 0,  3]])
kernalScharr_Y = np.array([[ -3,-10, -3],
                           [  0,  0,  0],
                           [  3, 10,  3]])
lenaScharr_X = ndimage.convolve(blured, kernalScharr_X)
lenaScharr_Y = ndimage.convolve(blured, kernalScharr_Y)

cv2.imshow("OG", lena)
cv2.imshow("Blured", blured)
cv2.imshow("Sobel_x", lenaSobel_X)
cv2.imshow("Sobel_y", lenaSobel_Y)
cv2.imshow("Laplacian", lenaLaplacian)
cv2.imshow("Scharr_x", lenaScharr_X)
cv2.imshow("Scharr_y", lenaScharr_Y)
cv2.waitKey()
cv2.destroyAllWindows()