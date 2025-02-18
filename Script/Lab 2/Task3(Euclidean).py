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

def euclidean_func(n, m):
    ctr = n // 2
    euclidean_map = np.zeros((n, m), dtype=np.uint16)

    for i in range(n):
        for j in range(m):
            euclidean_map[i, j] = np.sqrt(np.power((i - ctr), 2) + np.power((j - ctr), 2))
    answer = (euclidean_map / euclidean_map.max()) * 255
    return answer.astype(np.uint8)


out = euclidean_func(501,501)

cv2.imshow('Downscale 16', downscale16(out, 16))
cv2.waitKey()

cv2.imshow('Downscale 16', downscale16(out, 4))
cv2.waitKey()
cv2.imshow('Downscale 16', downscale16(out, 2))
cv2.waitKey()
cv2.imshow('Downscale 16', downscale16(out, 1))
cv2.waitKey()
