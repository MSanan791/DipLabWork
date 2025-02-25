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

img1 = readimg_grey(r"D:\Semester 6\DIP\Lab\Lab 5\lena.png")

img1_mean = np.mean(img1)
