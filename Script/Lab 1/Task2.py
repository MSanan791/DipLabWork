import numpy as np
import cv2

my_zero_image = np.zeros((500,500,1), dtype=(np.uint8))
my_zero_image[0+10:510-10,0+10:510-10] = 250

cv2.imshow( 'Win', my_zero_image)
cv2.waitKey()
