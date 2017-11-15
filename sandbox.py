import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./img/bolts2.jpg',0)
small_kernel = np.ones((3,3),np.uint8)
med_kernel = np.ones((5,5),np.uint8)
large_kernel = np.ones((9,9),np.uint8)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, small_kernel)
erosion = cv2.erode(gradient,med_kernel,iterations = 1)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(erosion,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
dilation = cv2.dilate(th3,large_kernel,iterations = 2)

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.imshow('Original',img)
cv2.namedWindow('Gradient', cv2.WINDOW_NORMAL)
cv2.imshow('Gradient',gradient)
cv2.namedWindow('Thresh', cv2.WINDOW_NORMAL)
cv2.imshow('Thresh',th3)
cv2.namedWindow('Dilation', cv2.WINDOW_NORMAL)
cv2.imshow('Dilation',dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.subplot(121),plt.imshow(img,cmap = 'grey')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(gradient,cmap = 'grey')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#
# plt.show()
