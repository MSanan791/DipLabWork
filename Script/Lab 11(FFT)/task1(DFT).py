
import numpy as np
import matplotlib.pyplot as plt
import cv2

def readimg_greyscale(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def showimg(img, title="Image"):
    cv2.imshow(title, img)
    return


def fourier_transform(img2, shift=0):
    img = img2.copy().astype(np.float64)


    ox, oy = img.shape
    img = cv2.resize(img, (64, 64))

    fft_img = np.zeros(img.shape, dtype=np.complex128)
    x, y = fft_img.shape

    # Pre-shift the image to center
    for i in range(x):
        for j in range(y):
            img[i, j] *= (-1) ** (i + j)

    # Manual DFT
    for i in range(x):
        for j in range(y):
            sum_val = 0
            for k in range(x):
                for l in range(y):
                    sum_val += img[k, l] * np.exp(-2j * np.pi * ((i * k / x) + (j * l / y)))
            fft_img[i, j] = sum_val

    mag_fft = np.abs(fft_img)
    mag_fft = np.log(mag_fft + 1)
    mag_fft = np.fft.fftshift(mag_fft)  # center the output
    img_8bit = cv2.normalize(mag_fft, None, 0, 255, cv2.NORM_MINMAX)
    img_8bit = img_8bit.astype(np.uint8)
    img_8bit = cv2.resize(img_8bit, (oy, ox))

    return img_8bit


def fft_builtin(img):
    img_fft = img.copy()

    img_fft = np.fft.fft2(img_fft)
    img_fft = np.abs(img_fft)
    img_fft = np.fft.fftshift(img_fft)
    img_fft = np.log(img_fft + 1)
    img_fft = cv2.normalize(img_fft, None, 0, 255, cv2.NORM_MINMAX)
    img_fft = img_fft.astype(np.uint8)


    return img_fft

img = readimg_greyscale(r'./Fig01 (1).tif')

fft_img = fourier_transform(img)

# DO NOT re-read img again
# fft_built = fft_builtin(fft_img)  # WRONG
fft_built = fft_builtin(img)  # CORRECT

plt.subplot(1, 2, 1)
plt.title("DFT (Manual)")
plt.imshow(fft_img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("DFT (NumPy)")
plt.imshow(fft_built, cmap='gray')

plt.show()

