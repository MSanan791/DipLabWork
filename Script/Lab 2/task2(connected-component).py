import numpy as np
import cv2

def connected_component(image):
    [x, y] = image.shape
    dic = []
    rem = []
    val = int(20)
    new_image = np.zeros((x+2, y+2), np.uint8)
    new_image[1:x+1,1:y+1] = image
    objects = np.zeros((x, y), np.uint8)

    for i in range(1,x+1):
        for j in range(1,y+1):
            if(new_image[i][j] >= 5 ):
                if (new_image[i-1][j] != 0 or new_image[i][j-1] != 0):
                    if(new_image[i-1][j] != 0 and new_image[i][j-1] != 0):
                        if new_image[i][j - 1] in dic:
                            dic.remove(new_image[i][j - 1])

                        rem.append(new_image[i][j-1])
                        objects[i-1][j-1] = new_image[i][j-1]
                    else:
                        objects[i - 1][j - 1] = val
                else:
                    val = val + int(40)
                    dic.append(val)
                    objects[i-1][j-1] = val
    for k in range(len(rem)-1,-1):
        for i in range(0,x):
            for j in range(0,y):
                if(objects[i][j] == rem[k]):


                    objects[i][j] = dic[len(dic)-1]
        dic.pop()
    return objects







image_data = cv2.imread(r"C:\Users\sanan\Pictures\cc.png", cv2.IMREAD_GRAYSCALE)
image = np.array(image_data)
cv2.imshow("new Image", connected_component(image))
cv2.waitKey()

