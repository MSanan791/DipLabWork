import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def PDF(image_hist, image):
    [x,y] = image.shape
    PDF = np.zeros(256)
    for i in range(256):
        PDF[i] = image_hist[i] / (x*y)
    return PDF

def CPDF(image_hist, image):
    [x,y] = image.shape
    CPDF = np.zeros(256)
    pdf = 0
    for i in range(256):
        pdf += image_hist[i] / (x*y)
        CPDF[i] = pdf
    return CPDF

def transform_f(image, transform_funct):
    [x,y] = image.shape
    for i in range(x):
        for j in range(y):
            image[i,j] = transform_funct[image[i,j]]
    return image

image_data = cv.imread(r"low_con.jpg", cv.IMREAD_GRAYSCALE )
image = np.array(image_data)

image_hist = np.zeros(256)

[x,y] = image.shape

for i in range(x):
    for j in range(y):
        image_hist[image[i,j]] += 1

plt.plot(range(256),image_hist)
plt.show()
PDF = PDF(image_hist, image)
plt.plot(range(256),PDF)
plt.show()
CPDF = CPDF(image_hist, image)
plt.plot(range(256),CPDF)
plt.show()
transform_funct = CPDF * 255
plt.plot(range(256),transform_funct)
plt.show()

cv.imshow("image", image)
cv.waitKey(0)
cv.imshow("image_hist", transform_f(image, transform_funct))
cv.waitKey(0)

# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
#
#
# def compute_pdf(image_hist, image_size):
#     return image_hist / image_size
#
#
# def compute_cpdf(image_hist, image_size):
#     return np.cumsum(image_hist) / image_size
#
#
# def transform_image(image, transform_funct):
#     transformed_image = np.zeros_like(image)
#     for i in range(3):  # Loop through R, G, B channels
#         transformed_image[:, :, i] = transform_funct[i][image[:, :, i]]
#     return transformed_image
#
#
# # Load image in color
# image_data = cv.imread(r"C:\Users\usama\Pictures\DSC01129.JPG")
#
#
# image = np.array(image_data)
#
# # Get image dimensions
# x, y, c = image.shape
# image_size = x * y
#
# # Initialize histograms for R, G, B
# image_hist = [np.zeros(256) for _ in range(3)]
#
# # Compute histograms for each channel
# for i in range(3):  # Loop through R, G, B
#     image_hist[i], _ = np.histogram(image[:, :, i], bins=256, range=(0, 256))
#
# # Plot histograms
# for i, color in enumerate(['r', 'g', 'b']):
#     plt.plot(range(256), image_hist[i], color=color)
# plt.title("Color Channel Histograms")
# plt.show()
#
# # Compute PDFs and CPDFs
# pdfs = [compute_pdf(image_hist[i], image_size) for i in range(3)]
# cpdfs = [compute_cpdf(image_hist[i], image_size) for i in range(3)]
#
# # Compute transformation functions
# transform_funct = [cpdfs[i] * 255 for i in range(3)]
#
# # Apply transformation
# transformed_image = transform_image(image, transform_funct)
# transformed_image = cv.medianBlur(transformed_image, 15)
# sharpening_kernel = np.array([[0, -1,  0],
#                                [-1,  5, -1],
#                                [0, -1,  0]])
#
# # Apply the sharpening filter
# transformed_image = cv.filter2D(transformed_image, -1, sharpening_kernel)
#
# lab_image = cv.cvtColor(transformed_image, cv.COLOR_BGR2LAB)
#
# # Split LAB channels
# l, a, b = cv.split(lab_image)
#
# # Apply CLAHE (Adaptive Histogram Equalization) to L-channel
# clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# l = clahe.apply(l)
#
# # Merge LAB channels back
# lab_image = cv.merge([l, a, b])
#
# # Convert back to BGR color space
# transformed_image = cv.cvtColor(lab_image, cv.COLOR_LAB2BGR)
#
# # Save the equalized image
# cv.imwrite(r"Equalized_Image.JPG", transformed_image)
#
# # Show original and transformed images
# cv.imshow("Original Image", image)
# cv.imshow("Equalized Image", transformed_image)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
#
