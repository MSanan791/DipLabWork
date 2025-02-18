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

def power_transform(img, gamma):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            img[i][j] = pow(img[i][j]/255, gamma) * 255
    return img




img1 = readimg_grey(r"D:\Semester 6\DIP\Lab\Lab 3\document.jpg")
# showing original imagee
cv.imshow("img1", img1)
cv.waitKey(0)

# applying power transformation
img2 = power_transform(img1, 4)
cv.imshow("img2", img2)
cv.waitKey(0)

[x,y] = img2.shape
img2_cropped = img2[150:x - 150, 100:y - 100]  # Crop 500 pixels from height and width


cv.imshow("img2_n", img2_cropped)
cv.waitKey(0)
[x, y] = img2_cropped.shape

for i in range(x):
    for j in range(y):
        if (img2_cropped[i][j] < 25):
            img2_cropped[i][j] = 0
        else:
            img2_cropped[i][j] = 255

cv.imshow("img2_n", img2_cropped)
cv.waitKey(0)
