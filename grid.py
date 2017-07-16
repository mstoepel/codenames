import cv2
import numpy as np

img = cv2.imread('Images\image2.JPG')
# im_color = cv2.applyColorMap(img,cv2.COLORMAP_WINTER)

im_color = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# BLUE ONLY

lower_blue = np.array([75,50,50])
upper_blue = np.array([130,255,255])

mask = cv2.inRange(im_color, lower_blue, upper_blue)
res = cv2.bitwise_and(img,img, mask=mask)
cv2.imshow('mapped_image',res)
cv2.waitKey(0)

# RED ONLY

lower_red = np.array([150,150,150])
upper_red = np.array([255,255,255])

mask2 = cv2.inRange(im_color, lower_red, upper_red)
res2 = cv2.bitwise_and(img,img,mask=mask2)
cv2.imshow('mapped_image',res2)
cv2.waitKey(0)

# BOTH

lower = np.array([50,50,150])
upper = np.array([255,255,255])

mask3 = cv2.inRange(im_color, lower, upper)
res3 = cv2.bitwise_and(img,img,mask=mask3)
cv2.imshow('mapped_image',res3)
cv2.waitKey(0)
