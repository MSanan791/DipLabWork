import numpy as np
import cv2


def erosion(structuring_element, picture1_element):
    picture_element = picture1_element.copy()
    x, y = picture_element.shape
    sx, sy = structuring_element.shape
    lr, tb = sx // 2, sy // 2
    pad_picture_element =  np.pad(picture_element, ((lr, lr), (tb, tb)), 'constant')
    for i in range(x):
        for j in range(y):
            matrixofinterest = pad_picture_element[i:i+sx, j:j+sy]
            erode = False
            if picture_element[i, j] >= 20:
                boolmatrix = matrixofinterest >= structuring_element
                erode = np.any(boolmatrix == False)
            if(erode == True):
                picture_element[i, j] = np.min(matrixofinterest)

    return picture_element

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
                picture_element[i, j] = np.max(matrixofinterest)
    return picture_element


img = cv2.imread(r'./Fig01.tif', cv2.IMREAD_GRAYSCALE)

erode = erosion(cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)), img)
dilate = dilation(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), img)
cv2.imshow('Erosion', erode)
cv2.imshow('Dilation', dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()