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

def less_mean(img, mean):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            if(img[i][j] > mean):
                img[i][j] = 255
            else:
                img[i][j] = 0
    cv.imshow('img', img)
    cv.waitKey(0)
def great_mean(img, mean):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            if(img[i][j] > mean):
                img[i][j] = 0
            else:
                img[i][j] = 255
    cv.imshow('img', img)
    cv.waitKey(0)

def twenty_mean(img, mean):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            if(img[i][j] >= mean - 20  and img[i][j] <= mean + 20):
                img[i][j] = 0
            else:
                img[i][j] = 255
    cv.imshow('img', img)
    cv.waitKey(0)

img1 = readimg_grey(r"gradient.png")
img1_mean = np.mean(img1)

less_mean(img1, img1_mean)
img1 = readimg_grey(r"gradient.png")
img1_mean = np.mean(img1)
great_mean(img1, img1_mean)
img1 = readimg_grey(r"gradient.png")
img1_mean = np.mean(img1)
twenty_mean(img1, img1_mean)


