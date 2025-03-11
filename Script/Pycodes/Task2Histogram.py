import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def compute_histogram(image):
    """ Compute histogram manually without using OpenCV's built-in function. """
    hist = np.zeros(256, dtype=np.int32)
    for pixel in image.ravel():
        hist[pixel] += 1
    return hist


def compute_pdf(hist, total_pixels):
    """ Compute the Probability Density Function (PDF). """
    return hist / total_pixels


def compute_cpdf(pdf):
    """ Compute the Cumulative Probability Density Function (CPDF). """
    return np.cumsum(pdf)


def histogram_equalization(image):
    """ Perform histogram equalization manually. """
    hist = compute_histogram(image)
    total_pixels = image.size
    pdf = compute_pdf(hist, total_pixels)
    cpdf = compute_cpdf(pdf)

    # Mapping function
    transform_funct = (cpdf * 255).astype(np.uint8)

    # Apply transformation using NumPy
    equalized_image = transform_funct[image]

    return equalized_image, hist, transform_funct


# Load grayscale image
image = cv.imread("low_con.jpg", cv.IMREAD_GRAYSCALE)  # Adjust path if needed

if image is None:
    print("Error: Image not found!")
    exit()

# Perform histogram equalization
equalized_image, hist, transform_funct = histogram_equalization(image)

# Plot original histogram
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(hist, color='black')
plt.title("Original Histogram")

# Plot transformation function
plt.subplot(1, 2, 2)
plt.plot(transform_funct, color='blue')
plt.title("Transformation Function")
plt.show()

# Display images
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap="gray")
plt.title("Histogram Equalized Image")
plt.axis("off")
plt.show()

# Save the processed image
cv.imwrite("/home/pi/equalized_image.jpg", equalized_image)
print("Equalized image saved as 'equalized_image.jpg'")

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
