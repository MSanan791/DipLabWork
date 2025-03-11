import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def local_threshholding(img):
    x, y = img.shape
    neighbourhood = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
    result = img.copy()

    for i in range(x):
        for j in range(y):
            kernel_f = []
            for nx, ny in neighbourhood:
                dx, dy = i + nx, j + ny
                if 0 <= dx < x and 0 <= dy < y:
                    kernel_f.append(img[dx, dy])
            mean = np.mean(kernel_f)
            if img[i, j] > mean:
                result[i, j] = 255
            else:
                result[i, j] = 0
    return result


def local_threshholding_median(img):
    x, y = img.shape
    neighbourhood = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
    result = img.copy()

    for i in range(x):
        for j in range(y):
            kernel_f = []
            for nx, ny in neighbourhood:
                dx, dy = i + nx, j + ny
                if 0 <= dx < x and 0 <= dy < y:
                    kernel_f.append(img[dx, dy])
            median = np.median(kernel_f)
            if img[i, j] > median:
                result[i, j] = 255
            else:
                result[i, j] = 0
    return result


file_name = r"Threshold_Image.png"
image_test = cv.imread(file_name, cv.IMREAD_GRAYSCALE)
plt.subplot(1, 2, 1)
plt.imshow(local_threshholding(image_test), cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(local_threshholding_median(image_test), cmap='gray')
plt.show()