import numpy as np
import cv2 as cv
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

def readimg_grey(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img_data = np.array(img)
    return img_data

def padder(img, pad_size):
    return np.pad(img, pad_size, 'constant')

def normalise_image(img):
    img = img / np.max(img)
    img = img * 255
    return img.astype(np.uint8)

def Laplacian(img):
    f_matrix = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    row , col = img.shape
    img_laplacian = np.zeros_like(img)
    pad_img = padder(img, 1)

    for i in range(0,row):
        for j in range(0,col):
            neighbourhood = pad_img[i:i+3,j:j+3]
            img_laplacian[i,j] = np.sum(np.multiply(f_matrix,neighbourhood))
    img_laplacian = normalise_image(img_laplacian)
    cv.imshow("Laplacian", img_laplacian)
    return img_laplacian

file_name = r"Fig03.tif"
img = readimg_grey(file_name)
cv.imshow("Original", img)

img2 = img + Laplacian(img)
cv.imshow("filtered Laplacian", img2)
cv.waitKey(0)