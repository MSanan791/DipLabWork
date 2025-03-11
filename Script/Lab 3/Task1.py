import numpy as np
import cv2 as cv

def readimg_grey(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img_data = np.array(img)
    return img_data
def readimg_color(path):
    img = cv.imread(path)
    img_data = np.array(img)
    return img_data

img1 = readimg_grey(r"gradient.png")
img1_mean = np.mean(img1)

[x,y] = img1.shape

for i in range(x):
    for j in range(y):
        if img1[i,j] > img1_mean:
            img1[i,j] = 255
        else:
            img1[i,j] = 0

cv.imshow("img1", img1)
cv.waitKey(0)

img1 = readimg_grey(r"gradient.png")
[x,y] = img1.shape
for i in range(x):
    for j in range(y):
        img1[i,j] = 255 - img1[i,j]

cv.imshow("img1", img1)
cv.waitKey(0)