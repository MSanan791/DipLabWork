import numpy as np
import cv2 as cv




img =  cv.imread(r"C:\Users\usama\Pictures\Whats app\IMG-20250223-WA0041[1].jpg", cv.IMREAD_GRAYSCALE)

[x,y] = img.shape

for i in range(x):
    for j in range(y):
        if img[i,j] > 210:
            img[i,j] = 255
        else:
            img[i,j] = 0

cv.imshow("img", img)
cv.waitKey(0)

cv.imwrite("img2.jpg", img)