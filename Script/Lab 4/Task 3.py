# %%
import numpy as np
import cv2 as cv

def mask_img(image):
    mask = np.ones((3,3,1), dtype=np.uint8)/9
    x,y = image.shape
    padd_img = pad_img(image)

    for i in range(1,x+1):
        for j in range(1,y+1):
            padd_img[i,j] = np.sum(mask[0:2, 0:2] * padd_img[i-1:i+1,j-1:j+1])
    padd_img = unpad_img(padd_img)
    return padd_img

def pad_img(image):
    x, y = image.shape
    x_mask = x+2
    y_mask = y+2
    padded_image = np.zeros((x_mask, y_mask), dtype=np.uint8)
    padded_image[1:x+1, 1:y+1] = image
    return padded_image

def unpad_img(image):
    x, y = image.shape
    x_mask = x-2
    y_mask = y-2
    unpadded_image = image[1:x_mask+1, 1:y_mask+1]
    return unpadded_image
# %%
img = cv.imread(r"fig05.tif", cv.IMREAD_GRAYSCALE)
img_data = np.array(img)
cv.imshow("img1", img_data)
cv.waitKey(0)
mask = mask_img(img_data)
cv.imshow("img1", mask)
cv.waitKey(0)