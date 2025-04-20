from pickletools import uint8

import numpy as np
import cv2 as cv

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def dice_coef(mask1, mask2):
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    intersection = np.sum(mask1 * mask2)
    dice = 2.*(intersection/(np.sum(mask1) + np.sum(mask2) +1e-6))
    return dice

def connected_component(image, connectivity=8):
    """
    Performs connected component labeling on binary image
    Args:
        image: Input binary image (numpy array)
        connectivity: 8 or 4 for 8-way or 4-way connectivity
    Returns:
        Labeled image with unique integers for each component
    """
    # Input validation
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Pad image to handle border pixels
    padded = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    [height, width] = padded.shape
    labels = np.zeros_like(padded, dtype=np.int32)

    # First pass: initial labeling
    current_label = 1
    label_equivalences = {}  # Store label equivalences

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if padded[i, j] >= 5:  # Assuming threshold of 5 as in original code
                # Get neighboring labels
                neighbors = []
                neighbors.append(labels[i - 1, j])  # North
                neighbors.append(labels[i, j - 1])  # West
                if connectivity == 8:
                    neighbors.append(labels[i - 1, j - 1])  # Northwest
                    neighbors.append(labels[i - 1, j + 1])  # Northeast

                # Remove zero labels
                neighbors = [n for n in neighbors if n > 0]

                if not neighbors:  # No neighbors have labels
                    labels[i, j] = current_label
                    current_label += 1
                else:
                    # Assign smallest neighbor label
                    min_label = min(neighbors)
                    labels[i, j] = min_label

                    # Record label equivalences
                    for n in neighbors:
                        if n != min_label:
                            if min_label not in label_equivalences:
                                label_equivalences[min_label] = set()
                            label_equivalences[min_label].add(n)

    # Resolve label equivalences
    final_labels = np.arange(current_label, dtype=np.int32)
    for label in label_equivalences:
        connected_labels = label_equivalences[label]
        for connected in connected_labels:
            final_labels[connected] = final_labels[label]

    # Second pass: update labels
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if labels[i, j] > 0:
                labels[i, j] = final_labels[labels[i, j]]

    # Return unpadded result
    return labels[1:-1, 1:-1]


# Example usage:
def segment_image(image_path):
    # Read image
    image = image_path
    if image is None:
        raise ValueError("Could not read image")

    # Apply connected component labeling
    labels = connected_component(image)

    # Visualize results (normalize labels for display)
    normalized_labels = (255 * labels / labels.max()).astype(np.uint8)

    # Display results
    # cv.imshow("Original Image", image)
    # cv.imshow("Connected Components", normalized_labels)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return labels

def power_transform(img, gamma):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            img[i][j] = pow(img[i][j]/255, gamma) * 255
    cv.imshow('img', img)
    cv.waitKey(0)

def log_transform(img):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            img[i][j] = np.log10(1 + img[i][j]) * 255
    cv.imshow('img', img)
    cv.waitKey()

def thresh1(img):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            if img[i][j] > 180:
                img[i][j] = 255
            else:
                img[i][j] = 0
    # cv.imshow('img', img)
    return img

def Background(img):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            if img[i][j] > 1:
                img[i][j] = 0
            else:
                img[i][j] = 50
    # cv.imshow('img', img)
    return img


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



img1 = cv.imread(r'.\dataset\images\render0001.png', cv.IMREAD_GRAYSCALE)

img2 = img1.copy()
cv.equalizeHist(img2)
Background1 = Background(img1)




bigRocks = thresh1(img2)
bigRocks2 = bigRocks.copy()
bigRocks = segment_image(bigRocks)

[x, y] = bigRocks.shape

for i in range(x):
    for j in range(y):
        if bigRocks[i, j] > 10:
            bigRocks[i, j] = 255

bigRocks2 = bigRocks2 - bigRocks

for i in range(x):
    for j in range(y):
        if bigRocks2[i, j] > 10:
            bigRocks2[i, j] = 20



plt.title('Big Rocks')
plt.imshow(bigRocks2, cmap="gray")
plt.show()
plt.figure()
plt.title('Gravel')
plt.imshow(bigRocks, cmap="gray")
plt.show()
plt.figure()
plt.title('Background')
plt.imshow(Background1, cmap="gray")
plt.show()
plt.axis("off")

cv.imshow('img1', Background1)

final = concat_image(bigRocks2, Background1)
final = concat_image(final, bigRocks2)

plt.figure()
plt.title('Final')
plt.imshow(final, cmap="gray")
plt.show()
plt.axis("off")

cv.imshow('img1', Background1)
mask = cv.imread(r".\dataset\masks\clean0001.png", cv.IMREAD_GRAYSCALE)
dicecoefficient = dice_coef(final, mask )
print('Dice coefficient:', dicecoefficient)
cv.waitKey(0)









