import numpy as np
import cv2 as cv
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
def readimg_grey(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img_data = np.array(img)
    return img_data

def readimg_color(path):
    img = cv.imread(path)
    img_data = np.array(img)
    return img_data

def power_transform(img, gamma):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            img[i][j] = pow(img[i][j]/255, gamma) * 255
    cv.imshow('img', img)
    cv.waitKey(0)

def log_transform(img):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            img[i][j] = np.log10(1 + img[i][j]) * 255
    cv.imshow('img', img)
    cv.waitKey()

img1 = readimg_grey(r"fig01.tif")
power_transform(img1, 0.2)
img1 = readimg_grey(r"fig01.tif")
power_transform(img1, 0.5)
img1 = readimg_grey(r"fig01.tif")
power_transform(img1, 1.2)
img1 = readimg_grey(r"fig01.tif")
power_transform(img1, 1.8)
img1 = readimg_grey(r"fig01.tif")
log_transform(img1)

img1 = readimg_grey(r"D:\Semester 6\DIP\Lab\Lab 3\fig02.tif")
power_transform(img1, 0.2)
img1 = readimg_grey(r"D:\Semester 6\DIP\Lab\Lab 3\fig02.tif")
power_transform(img1, 0.5)
img1 = readimg_grey(r"D:\Semester 6\DIP\Lab\Lab 3\fig02.tif")
power_transform(img1, 1.2)
img1 = readimg_grey(r"D:\Semester 6\DIP\Lab\Lab 3\fig02.tif")
power_transform(img1, 1.8)
img1 = readimg_grey(r"D:\Semester 6\DIP\Lab\Lab 3\fig02.tif")
log_transform(img1)

img1 = readimg_grey(r"D:\Semester 6\DIP\Lab\Lab 3\fig03.tif")
power_transform(img1, 0.2)
img1 = readimg_grey(r"D:\Semester 6\DIP\Lab\Lab 3\fig03.tif")
power_transform(img1, 0.5)
img1 = readimg_grey(r"D:\Semester 6\DIP\Lab\Lab 3\fig03.tif")
power_transform(img1, 1.2)
img1 = readimg_grey(r"D:\Semester 6\DIP\Lab\Lab 3\fig03.tif")
power_transform(img1, 1.8)
img1 = readimg_grey(r"D:\Semester 6\DIP\Lab\Lab 3\fig03.tif")
log_transform(img1)

log_transform(img1)

