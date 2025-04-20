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

def Opening(structuring_element, picture1_element):
    processing_image = picture1_element.copy()
    processing_image = erosion(structuring_element, processing_image)
    processing_image = dilation(structuring_element, processing_image)
    return processing_image

def Closing(structuring_element, picture1_element):
    processing_image = picture1_element.copy()
    processing_image = dilation(structuring_element, processing_image)
    processing_image = erosion(structuring_element, processing_image)
    return processing_image




img= cv2.imread(r'./rice.png', cv2.IMREAD_GRAYSCALE)
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
opened = Opening(ellipse,img)
tophat =   opened - img

cv2.imshow('original', img)
cv2.imshow('opened', opened)
cv2.imshow('tophat', tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()
