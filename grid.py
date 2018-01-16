import cv2
import numpy as np
import random
import sys
from PIL import Image, ImageDraw

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
cv2.imwrite('out_test2.png',res3)
scipy.misc.imsave('out_test2.png',res3)


#### CV2 SCANNING 1

im = Image.open('grid_example.png')
isize = im.size
blue = Image.open('blue.png')
bsize = blue.size
x0,y0 = bsize[0] //2, bsize[1] //2
pixel = blue.getpixel((x0,y0))[:-1]

def diff(a,b):
    return sum((a-b)**2 for a, b in zip(a, b))

best = (100000,0,0)
for x in range(isize[0]):
    for y in range(isize[1]):
        ipixel = im.getpixel((x,y))
        d = diff(ipixel,pixel)
        if d < best[0]: best = (d,x,y)

draw = ImageDraw.Draw(im)
x,y = best[1:]
draw.rectangle((x-x0, y-y0, x+x0, y+y0))
print(best)
im.save('out.png')

#### CV2 SCANNING 2

im, pattern, samples = 'grid_example.png', 'blue.png', 15
samples = int(samples)

im = Image.open(im)
walnut = Image.open(pattern)
pixels = []
while len(pixels) < samples:
    x = random.randint(0, walnut.size[0] - 1)
    y = random.randint(0, walnut.size[1] - 1)
    pixel = walnut.getpixel((x, y))
    if pixel[-1] > 200:
        pixels.append(((x, y), pixel[:-1]))

def diff(a, b):
    return sum((a - b) ** 2 for a, b in zip(a, b))

best = []

for x in range(im.size[0]):
    for y in range(im.size[1]):
        d = 0
        for coor, pixel in pixels:
            try:
                ipixel = im.getpixel((x + coor[0], y + coor[1]))
                d += diff(ipixel, pixel)
            except IndexError:
                d += 256 ** 2 * 3
        # if the center of best isn't already captured
        best.append((d, x, y))
        best.sort(key = lambda x: x[0])
        print(best)
        best = best[:8]

draw = ImageDraw.Draw (im)
for best in best:
    x, y = best[1:]
    draw.rectangle((x, y, x + walnut.size [0], y + walnut.size[1]), outline = 'red')
im.save('out.png')


##### CV2 BF AND FLANN

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('grid_example.png',1)
img2 = cv2.imread('blue.png',1)

# BF

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50], None, flags=2)
plt.imshow(img3),plt.show()

# FLANN

orb = cv2.ORB_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()