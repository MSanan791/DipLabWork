import numpy as np
import cv2

array = np.zeros((510,510), np.uint8 )

for i in range(0,510,40):

        array[0:510, i:i+20] = 255

cv2.imshow( 'Win', array)
cv2.waitKey()


array = np.zeros((510,510), np.uint8 )
array[100:410, 100:410] = 255

cv2.imshow( 'Win', array)
cv2.waitKey()

array = np.zeros((510,510), np.uint8 )
array[0:510] = 255

for i in range(0,510,40):

        array[0:510, i:i+20] = 0
for i in range(0,510,40):

        array[i:i+20, 000:510 ] = 0

cv2.imshow( 'Win', array)
cv2.waitKey()