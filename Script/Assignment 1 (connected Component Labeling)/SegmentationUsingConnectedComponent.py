import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def dice_coef(mask1, mask2):
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    intersection = np.sum(mask1 * mask2)
    dice = 2.*(intersection/(np.sum(mask1) + np.sum(mask2) +1e-6))
    return dice

def thresh1(img):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            if img[i][j] > 85:
                img[i][j] = 0
            else:
                img[i][j] = 255

    return img

def thresh2(img):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            if img[i][j] > 180:
                img[i][j] = 0
            else:
                img[i][j] = 100

    return img


def connected_components(img):
    # Padding the image with zeros
    pad_img = np.pad(img, pad_width=1)
    img_out = np.zeros_like(pad_img, dtype=np.int32)  # Make output same size as padded image
    rows, cols = pad_img.shape

    # 8-connectivity neighbor offsets
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]  # Only need previous neighbors
    eq_labels = {}  # Dictionary to store equivalent labels
    current_label = 60  # Start labeling from 1

    # First Pass: Assign initial labels
    for i in range(1, rows - 1):  # Skip padding
        for j in range(1, cols - 1):  # Skip padding
            if pad_img[i, j] == 0:  # Skip background pixels
                continue

            neighbor_labels = []
            for dx, dy in neighbours:
                ni, nj = i + dx, j + dy
                if img_out[ni, nj] > 0:  # If neighbor has a label
                    neighbor_labels.append(img_out[ni, nj])

            if not neighbor_labels:  # No neighbors have labels
                img_out[i, j] = current_label
                eq_labels[current_label] = current_label
                current_label += 10
            else:
                min_label = min(neighbor_labels)
                img_out[i, j] = min_label

                # Update equivalences
                for label in neighbor_labels:
                    eq_labels[label] = min_label

    # Flatten equivalences
    for label in eq_labels:
        root = label
        while eq_labels[root] != root:
            root = eq_labels[root]
        eq_labels[label] = root

    # Second Pass: Resolve labels
    result = np.zeros_like(img, dtype=np.int32)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if img_out[i, j] > 0:
                result[i - 1, j - 1] = eq_labels[img_out[i, j]]

    return result


def get_segment(img):
    [x, y] = img.shape
    segments = []
    for i in range(x):
        for j in range(y):
            if img[i, j] > 1:
                segments.append([i, j])
    return segments

def concat_image(img1, img2):
    max_value = np.max(img1)
    img1[img1 > 0] = max_value
    segments = get_segment(img1)
    # cv.imshow('img1', img2)
    for segment in segments:
        [i,j] = segment
        img2[i , j] = img1[i , j]
    return img2

# Read image
filename = r"C:\Users\usama\PycharmProjects\DipLabWork\Script\Assignment 1 (connected Component Labeling)\dataset_DIP_assignment\test\images\241.bmp"
mask_file = r"C:\Users\usama\PycharmProjects\DipLabWork\Script\Assignment 1 (connected Component Labeling)\dataset_DIP_assignment\test\masks\241.png"
img1 = cv.imread(filename, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(filename, cv.IMREAD_GRAYSCALE)
original = cv.imread(filename, cv.IMREAD_GRAYSCALE)
mask_img = cv.imread(mask_file, cv.IMREAD_GRAYSCALE)
# PLotting Histogram
image_hist = np.zeros(256)
[x,y] = original.shape
for i in range(x):
    for j in range(y):
        image_hist[original[i,j]] += 1
plt.plot(range(256),image_hist)
plt.show()
# Threshold the image
img1 = thresh1(img1)
img2 = thresh2(img2)



# Applying connected components
labeled = connected_components(img1)
# Normalize for visualization
# Scale the labels to be visible (multiply by a factor that spreads the colors)
max_label = np.max(labeled)
if max_label > 0:
    scaled = (labeled * (255 // max_label)).astype(np.uint8)
else:
    scaled = np.zeros_like(labeled, dtype=np.uint8)

img2 = concat_image(scaled, img2)


# plotting the images
plt.subplot(1, 3, 1)
plt.imshow(original, cmap='gray')
plt.title("Original")
plt.axis("off")

# Second image
plt.subplot(1, 3, 2)
plt.imshow(img2, cmap='gray')
plt.title("CCA Segmentation")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mask_img, cmap='gray')
plt.title("Mask")
plt.axis("off")

# Show the figure
plt.show()

score = dice_coef(img2, mask_img )


print(f"Score: {score}")

cv.destroyAllWindows()


