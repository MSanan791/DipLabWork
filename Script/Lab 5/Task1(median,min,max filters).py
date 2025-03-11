import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def readimg_grey(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img_data = np.array(img)
    return img_data
def readimg_color(path):
    img = cv.imread(path)
    img_data = np.array(img)
    return img_data

def filter_median(img1, size = 3):
    img = img1.copy()
    x,y = img.shape
    pad_img = img.copy()
    pad_img = np.pad(pad_img, (size//2, size//2), 'constant')
    for i in range(x):
        for j in range(y):
            neighbors = pad_img[i:i+size, j:j+size]
            median1 = np.median(neighbors.flatten())
            img[i,j] = median1

    return img

def filter_min(img1, size = 3):
    img = img1.copy()
    x,y = img.shape
    pad_img = img.copy()
    pad_img = np.pad(pad_img, (size//2, size//2), 'constant')
    for i in range(x):
        for j in range(y):
            neighbors = pad_img[i:i+size, j:j+size]
            median1 = np.min(neighbors.flatten())
            img[i,j] = median1

    return img

def filter_max(img1, size = 3):
    img = img1.copy()
    x , y = img.shape
    pad_img = img.copy()
    pad_img = np.pad(pad_img, (size//2, size//2), 'constant')
    for i in range(x):
        for j in range(y):
            neighbors = pad_img[i:i+size, j:j+size]
            median1 = np.max(neighbors.flatten())
            img[i,j] = median1
    return img

original = readimg_grey("F:Fig02.tif")
img1 = readimg_grey(r"Fig02.tif")
img2 = readimg_grey(r"Fig02.tif")


plt.subplot(2,4,1)
plt.imshow(img1, cmap = 'gray')
plt.title("Original")

plt.subplot(2,4,2)
plt.imshow(filter_median(img1,3), cmap = 'gray')
plt.title("Median 3x3")

plt.subplot(2,4,3)
plt.imshow(filter_median(img1,15), cmap = 'gray')
plt.title("Median 15x15")

plt.subplot(2,4,4)
plt.imshow(filter_median(img1,31), cmap = 'gray')
plt.title("Median 31x31")

plt.subplot(2,4,5)
plt.imshow(img2, cmap = 'gray')
plt.title("Original")

plt.subplot(2,4,6)
plt.imshow(filter_median(img2,3), cmap = 'gray')
plt.title("Median 3x3")

plt.subplot(2,4,7)
plt.imshow(filter_median(img2,15), cmap = 'gray')
plt.title("Median 15x15")

plt.subplot(2,4,8)
plt.imshow(filter_median(img2,31), cmap = 'gray')
plt.title("Median 31x31")

plt.figure()

plt.subplot(2,4,1)
plt.imshow(img1, cmap = 'gray')
plt.title("Original")

plt.subplot(2,4,2)
plt.imshow(filter_min(img1,3), cmap = 'gray')
plt.title("min 3x3")

plt.subplot(2,4,3)
plt.imshow(filter_min(img1,15), cmap = 'gray')
plt.title("min 15x15")

plt.subplot(2,4,4)
plt.imshow(filter_min(img1,31), cmap = 'gray')
plt.title("min 31x31")

plt.subplot(2,4,5)
plt.imshow(img2, cmap = 'gray')
plt.title("Original")

plt.subplot(2,4,6)
plt.imshow(filter_min(img2,3), cmap = 'gray')
plt.title("min 3x3")

plt.subplot(2,4,7)
plt.imshow(filter_min(img2,15), cmap = 'gray')
plt.title("min 15x15")

plt.subplot(2,4,8)
plt.imshow(filter_min(img2,31), cmap = 'gray')
plt.title("min 31x31")

plt.figure()

plt.subplot(2,4,1)
plt.imshow(img1, cmap = 'gray')
plt.title("Original")

plt.subplot(2,4,2)
plt.imshow(filter_max(img1,3), cmap = 'gray')
plt.title("max 3x3")

plt.subplot(2,4,3)
plt.imshow(filter_max(img1,15), cmap = 'gray')
plt.title("max 15x15")

plt.subplot(2,4,4)
plt.imshow(filter_max(img1,31), cmap = 'gray')
plt.title("max 31x31")

plt.subplot(2,4,5)
plt.imshow(img2, cmap = 'gray')
plt.title("Original")

plt.subplot(2,4,6)
plt.imshow(filter_max(img2,3), cmap = 'gray')
plt.title("max 3x3")

plt.subplot(2,4,7)
plt.imshow(filter_max(img2,15), cmap = 'gray')
plt.title("max 15x15")

plt.subplot(2,4,8)
plt.imshow(filter_max(img2,31), cmap = 'gray')
plt.title("max 31x31")


plt.show()

