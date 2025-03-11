import numpy as  np
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

def level_slicing(img, low_level, high_level, value):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            if(img[i][j] >= low_level and img[i][j] <= high_level):
                img[i][j] = value
    cv.imshow('img', img)
    cv.waitKey(0)



img1 = readimg_grey(r"gradient.png")
level_slicing(img1, 100,200, 210)