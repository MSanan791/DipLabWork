import numpy as np
import cv2

my_zero_image = np.zeros((510,510,1), dtype=(np.uint8))
my_zero_image[0+5:510-5,0+5:510-5] = 250

cv2.imshow( 'Win', my_zero_image)
cv2.waitKey()
