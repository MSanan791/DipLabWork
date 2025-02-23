import cv2 as cv
import numpy as np

def thresh1(img):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            if img[i][j] > 70:
                img[i][j] = 0
            else:
                img[i][j] = 255
    cv.imshow('img', img)
    cv.waitKey(0)
    return img

def thresh2(img):
    [x, y] = img.shape
    for i in range(x):
        for j in range(y):
            if img[i][j] > 180:
                img[i][j] = 0
            else:
                img[i][j] = 155
    cv.imshow('img', img)
    cv.waitKey(0)
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


# Read image
filename = r"D:\Semester 6\Projects\Pycharm\DIP\DIP Lab Work\Script\Assignment 1 (connected Component Labeling)\dataset_DIP_assignment\train\images\003.bmp"
img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
cv.imshow("Original", img)
cv.waitKey(0)
# Threshold the image if needed
thresh2(img)
# Apply connected components
labeled = connected_components(img)

# Normalize for visualization
# Scale the labels to be visible (multiply by a factor that spreads the colors)
max_label = np.max(labeled)
if max_label > 0:
    scaled = (labeled * (255 // max_label)).astype(np.uint8)
else:
    scaled = np.zeros_like(labeled, dtype=np.uint8)
cv.imshow("Connected Components", scaled)
cv.waitKey(0)
cv.destroyAllWindows()


