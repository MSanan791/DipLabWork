import numpy as np
import cv2

def gaussian_low_pass_filter(img, D0):
    P, Q = img.shape
    center_x, center_y = P // 2, Q // 2
    distance_map = np.zeros((P, Q), dtype=np.float32)

    for u in range(P):
        for v in range(Q):
            distance = np.sqrt((u - center_x) ** 2 + (v - center_y) ** 2)
            distance_map[u, v] = np.exp(-(distance ** 2) / (2 * (D0 ** 2)))

    return distance_map

img = cv2.imread('Fig01 (1).tif', cv2.IMREAD_GRAYSCALE)
glow = gaussian_low_pass_filter(img, 30)

fftimg = np.fft.fft2(img)
fftimg = np.fft.fftshift(fftimg)

img_filtered = fftimg * glow  # directly multiply

img_filtered = np.fft.ifftshift(img_filtered)
img_filtered = np.fft.ifft2(img_filtered)
img_filtered = np.abs(img_filtered)

img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)
img_filtered = img_filtered.astype(np.uint8)

cv2.imshow('Filtered Image', img_filtered)
cv2.imshow('Original Image', img)
cv2.imshow('Gaussian Low Pass Filter', glow)
cv2.waitKey(0)
cv2.destroyAllWindows()
