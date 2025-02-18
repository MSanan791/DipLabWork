import numpy as np
import cv2

def sheesha(img):
    [x,y,z] = img.shape
    x2 = int(x/2)
    y2 = int(y/2)

    for i in range (0,x2,1):
        img[x2+i,0:y] = img[x2-i, 0:y]

    return img

image = cv2.imread(r"C:\Users\sanan\Pictures\images.jpeg")
image_array =  np.array(image)


image_array = sheesha(image_array)

cv2.imshow( 'Win', image_array)
cv2.waitKey()