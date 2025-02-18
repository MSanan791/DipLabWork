import numpy as np
import cv2

def resize_image(original_image, scale_size, sample = 1):
    scale_size2 = int(scale_size/sample)
    [x, y, z] =original_image.shape
    test_i = np.zeros((scale_size2,scale_size2, z), np.uint8)
    new_width = min(x, scale_size2)
    new_height = min(y, scale_size2)

    test_i[0:new_width, 0:new_height] = original_image[0:new_width, 0:new_height]

    return test_i


image = cv2.imread(r"C:\Users\sanan\Pictures\images.jpeg")
image_array =  np.array(image)
cv2.imshow( 'Win', image_array)
cv2.waitKey()
# image_array = np.resize(image_array, (512,512))
image_array = resize_image(image_array,512,1)
image_array = resize_image(image_array,512,4)
cv2.imshow( 'Win', image_array)
cv2.waitKey()

cv2.imwrite('saved_image.jpg', image)
print('image have been saved')

