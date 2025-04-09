import numpy as np
import cv2
from matplotlib.pyplot import imshow


def dilation(structuring_element, picture1_element):
    picture_element = picture1_element.copy()
    x, y = picture_element.shape
    sx, sy = structuring_element.shape
    lr, tb = sx // 2, sy // 2
    pad_picture_element =  np.pad(picture_element, ((lr, lr), (tb, tb)), 'constant')
    for i in range(x):
        for j in range(y):
            matrixofinterest = pad_picture_element[i:i+sx, j:j+sy]
            dilate = False
            boolmatrix = (matrixofinterest!= 0) & (structuring_element == 1)
            dilate = np.any(boolmatrix == True)
            if(dilate == True):
                picture_element[i, j] = 255
    return picture_element

img = cv2.imread(r'./broken_text.tif', cv2.IMREAD_GRAYSCALE)
dil = dilation(cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), img)
dil = dilation(cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), img)
cv2.imshow('img', dil )


cv2.waitKey(0)
cv2.destroyAllWindows(0)
