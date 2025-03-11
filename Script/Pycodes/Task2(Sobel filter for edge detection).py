# %%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def sobel_filter(img):
    kernalx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    kernaly = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    rows, columns = img.shape
    sobel_x = np.zeros_like(img)
    sobel_y = np.zeros_like(img)

    pad_img = img.copy()
    pad_img = np.pad(pad_img, 1, 'constant')

    for i in range(0,rows):
        for j in range(0,columns):
            neighbourhood = pad_img[i:i+3, j:j+3]
            sobel_x[i, j] = np.sum(kernalx * neighbourhood)
            sobel_y[i, j] = np.sum(kernaly * neighbourhood)

    gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    gradient_phase = np.arctan2(sobel_y, sobel_x)

    [x,y] = sobel_x.shape
    sobeled = sobel_x + sobel_y
    sobeled = np.uint8(sobeled)

    sobel_x = normalise_image(sobel_x)
    sobel_y = normalise_image(sobel_y)
    sobeled = normalise_image(sobeled)
    gradient_magnitude = normalise_image(gradient_magnitude)
    gradient_phase = normalise_image(gradient_phase)
    cv.imshow("sobel_x", sobel_x)
    cv.imshow("sobel_y", sobel_y)
    cv.imshow('sobeled', sobeled)
    cv.imshow("gradient_magnitude", gradient_magnitude)
    cv.imshow("gradient_phase", gradient_phase)
    cv.waitKey(0)
    return gradient_magnitude, gradient_phase

def normalise_image(img):
    img = img / np.max(img)
    img = img * 255
    return img.astype(np.uint8)

f_name = r"Fig03.tif"
img = cv.imread(f_name, cv.IMREAD_GRAYSCALE)
sobel_filter(img)