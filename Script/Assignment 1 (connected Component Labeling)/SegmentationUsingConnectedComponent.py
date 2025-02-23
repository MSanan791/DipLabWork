import cv2 as cv
import numpy as np



def connected_components(img):
# padding current image with zeros
    pad_img = np.pad(img, pad_width=1)
    img_out = np.zeros_like(img)
# getting size of both images
    [x, y] = img.shape
    [x2, y2] = pad_img.shape
# Creating Labels
    neighbours = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1)]
    eq_labels = {} # dictionary to store labels
    value = 10 # start value of labels
# first iteration
    for i in range(x):
        i2 = i+1
        for j in range(y):
            j2 = j+1

            neighbor_labels = []
            for dx, dy in neighbours:
                ni, nj = i +dx, j+dy
                if 0 <= ni < x2 and 0 <= nj < y2 and pad_img[ni,nj] > 0:
                    neighbor_labels.append(pad_img(i,j))

            if not neighbor_labels:
                img_out[i,j] = value
                eq_labels[value] = value
                value = value + 1
            else:
                







    cv.imshow('img', pad_img)
    cv.waitKey(0)

    return


filenam = "F:\Other computers\My Laptop (1)\Semester 6\DIP\Lab\Lab 2\cc.png"
img = cv.imread(filenam, cv.IMREAD_GRAYSCALE)
connected_components(img)
