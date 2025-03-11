import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt




def global_threshholding(img):
    mean_global = np.mean(img)
    print(f'mean = {mean_global}')
    x , y= img.shape
    for i in range(x):
        for j in range(y):
            if(img[i][j] > mean_global):
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img

def median_threshholding(img):
    mean_median = np.mean(img)
    print(f'median = {mean_median}')
    x , y= img.shape
    for i in range(x):
        for j in range(y):
            if(img[i][j] > mean_median):
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img

file_name =r"Threshold_Image.png"
testing = cv.imread(file_name, cv.IMREAD_GRAYSCALE)
plt.subplot(2,2,1)
plt.imshow(testing, cmap='gray')
plt.subplot(2,2,2)
plt.imshow(global_threshholding(testing), cmap='gray')
plt.subplot(2,2,3)
plt.imshow(median_threshholding(testing), cmap='gray')
plt.show()
