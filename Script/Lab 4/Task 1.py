import cv2
import numpy as np
import cv2 as cv

img = cv2.imread(r"D:\Semester 6\DIP\Lab\Lab 4\low_con.jpg", cv.IMREAD_GRAYSCALE)
img_data = np.array(img)
[x, y] = img_data.shape

# Calculate percentiles once before the loop
p5 = np.percentile(img_data, 5)
p95 = np.percentile(img_data, 95)

cv.imshow("img1", img_data)
cv.waitKey(0)

for i in range(0, x):
    for j in range(0, y):
        if img_data[i, j] < p5:
            img_data[i, j] = 0
        elif img_data[i, j] > p95:
            img_data[i, j] = 255
        else:
            img_data[i, j] = 255 * (img_data[i, j] - p5) / (p95 - p5)

# Convert back to uint8 before displaying
img_data = img_data.astype(np.uint8)
cv.imshow("img1", img_data)
cv.waitKey(0)