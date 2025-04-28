import numpy as np
import cv2

def lowpass(img, d0):

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    d_map = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if distance > d0:
                d_map[i, j] = 0
            else:
                d_map[i, j] = 1

    return d_map

def highpass(img, d0):
    return 1 - lowpass(img, d0)

def filter_img(img, d0, filter_type='lowpass'):
    if filter_type == 'lowpass':
        mask = lowpass(img, d0)
    elif filter_type == 'highpass':
        mask = highpass(img, d0)
    else:
        raise ValueError("Invalid filter type. Choose 'lowpass' or 'highpass'.")

    # Apply the filter in the frequency domain
    img_fft = np.fft.fft2(img)
    img_fft_shifted = np.fft.fftshift(img_fft)
    filtered_img_fft = img_fft_shifted * mask
    filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered_img_fft))

    return np.abs(filtered_img)

img = cv2.imread('Fig01 (1).tif', cv2.IMREAD_GRAYSCALE)


img_l = filter_img(img, 400, 'lowpass')
img_h = filter_img(img, 400, 'highpass')

cv2.imshow('img', img)
cv2.imshow('Filtered Lowpass', img_l)
cv2.imshow('Filtered Highpass', img_h)
cv2.waitKey(0)
cv2.destroyAllWindows()