import numpy as np
import cv2

def make_img(pixelz = 500):
    img = np.zeros((pixelz,pixelz,3), np.uint8)
    img[0:pixelz] = 255
    img[0:int(pixelz/8),0:int(pixelz/8)] = 0
    img[0 : int(pixelz/8), 500-int(pixelz/8) : 500] = (255,0,0)
    img[500- int(pixelz/8) : 500, 0 : int(pixelz/8)] = (0,255,0)
    img[500 - int(pixelz / 8):500, 500- int(pixelz/8) : 500] = (0,0,255)

    return img

img = make_img()

cv2.imshow('WIN', img)
cv2.waitKey()