import numpy as np
import cv2

def resiz(a):
        test_a = np.zeros

    return


image = cv2.imread(r"C:\Users\sanan\Pictures\images.jpeg")
image_array =  np.array(image)

image_array = np.resize(image_array, (512,512))

cv2.imshow( 'Win', image_array)
cv2.waitKey()
