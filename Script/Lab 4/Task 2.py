import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def PDF(image_hist, image):
    [x,y] = image.shape
    PDF = np.zeros(256)
    for i in range(256):
        PDF[i] = image_hist[i] / (x*y)
    return PDF

def CPDF(image_hist, image):
    [x,y] = image.shape
    CPDF = np.zeros(256)
    pdf = 0
    for i in range(256):
        pdf += image_hist[i] / (x*y)
        CPDF[i] = pdf
    return CPDF

def transform_f(image, transform_funct):
    [x,y] = image.shape
    for i in range(x):
        for j in range(y):
            image[i,j] = transform_funct[image[i,j]]
    return image

image_data = cv.imread(r"D:\Semester 6\DIP\Lab\Lab 4\low_con.jpg", cv.IMREAD_GRAYSCALE )
image = np.array(image_data)

image_hist = np.zeros(256)

[x,y] = image.shape

for i in range(x):
    for j in range(y):
        image_hist[image[i,j]] += 1

plt.plot(range(256),image_hist)
plt.show()
PDF = PDF(image_hist, image)
plt.plot(range(256),PDF)
plt.show()
CPDF = CPDF(image_hist, image)
plt.plot(range(256),CPDF)
plt.show()
transform_funct = CPDF * 255
plt.plot(range(256),transform_funct)
plt.show()

cv.imshow("image", image)
cv.waitKey(0)
cv.imshow("image_hist", transform_f(image, transform_funct))
cv.waitKey(0)