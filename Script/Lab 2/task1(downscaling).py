import numpy as np
import cv2

def downscale16(image, divisor):
    factor = 256/divisor

    image2 = np.full_like(image, 0)

    for i in range(0,len(image)):
        for j in range(0,len(image[i])):
            image2[i][j] = int(image[i][j]/factor)
            image2[i][j] = image2[i][j] * factor
    return image2



image_load = cv2.imread(r"C:\Users\sanan\Pictures\gradient.png", cv2.IMREAD_GRAYSCALE)
image = np.array(image_load)


cv2.imshow('Downscale 16', downscale16(image, 16))
cv2.waitKey()

cv2.imshow('Downscale 16', downscale16(image, 4))
cv2.waitKey()
cv2.imshow('Downscale 16', downscale16(image, 2))
cv2.waitKey()
cv2.imshow('Downscale 16', downscale16(image, 1))
cv2.waitKey()
