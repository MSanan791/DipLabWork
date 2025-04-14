import numpy as np
import pandas as pd
import cv2 as cv


def white_padding(img, pad_pix):
    row = len(img)
    col = len(img[0])

    row_start = int(pad_pix / 2)
    col_start = int(pad_pix / 2)

    bordered_image = np.ones((row + pad_pix, col + pad_pix), dtype=np.uint8)
    bordered_image[0:row + pad_pix, 0:col + pad_pix] = 255
    bordered_image[row_start:row + row_start, col_start:col + col_start] = img[0:row, 0:col]

    return bordered_image


def horizontal_sobel_filter():
    filter_size = 3
    filter_mask = np.zeros((filter_size, filter_size), dtype=np.int16)
    filter_mask[0, 0] = -1
    filter_mask[0, 1] = -2
    filter_mask[0, 2] = -1
    filter_mask[1, 0] = 0
    filter_mask[1, 1] = 0
    filter_mask[1, 2] = 0
    filter_mask[2, 0] = 1
    filter_mask[2, 1] = 2
    filter_mask[2, 2] = 1

    return filter_mask


def vertical_sobel_filter():
    filter_size = 3
    filter_mask = np.zeros((filter_size, filter_size), dtype=np.int16)
    filter_mask[0, 0] = -1
    filter_mask[0, 1] = 0
    filter_mask[0, 2] = 1
    filter_mask[1, 0] = -2
    filter_mask[1, 1] = 0
    filter_mask[1, 2] = 2
    filter_mask[2, 0] = -1
    filter_mask[2, 1] = 0
    filter_mask[2, 2] = 1

    return filter_mask


def convolution(figure_patch, filter_mask):
    return np.sum(figure_patch * filter_mask)


def generic_function(figure, filter_mask, filter_size):
    rows, cols = figure.shape
    pad = filter_size // 2
    output = np.copy(figure)

    for i in range(pad, rows - pad):
        for j in range(pad, cols - pad):
            value = convolution(figure[i - pad:i + pad + 1, j - pad:j + pad + 1], filter_mask)
            output[i, j] = np.clip(value, 0, 255)

    return output


def normalize_manual(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val) * 255
    return normalized.astype(np.uint8)


def block_normalization(feature_vector, epsilon=1e-5, clip_threshold=0.2):
    squared_norm = np.sum(np.square(feature_vector)) + epsilon ** 2

    normalized_vector = feature_vector / np.sqrt(squared_norm)

    normalized_vector = np.clip(normalized_vector, None, clip_threshold)

    squared_norm_clipped = np.sum(np.square(normalized_vector)) + epsilon ** 2
    normalized_vector = normalized_vector / np.sqrt(squared_norm_clipped)

    return normalized_vector


def hog_transform(image):
    rows, cols = image.shape
    print("Image shape", rows, cols)

    block_size = 16
    cell_size = block_size // 2
    filter_size = 3
    no_blocks = rows // 16
    no_cell = 4
    bin_size = 6
    feature_vector_len = no_blocks * no_cell * bin_size
    print("Feature vector length", feature_vector_len)

    feature_vector = [0] * feature_vector_len

    horizontal_sobel = horizontal_sobel_filter()
    vertiacl_sobel = vertical_sobel_filter()

    padded_image = white_padding(image.copy(), filter_size - 1)
    cv.imshow("Padded Image ", padded_image)
    cv.waitKey(0)

    sobel_x = generic_function(padded_image.copy(), horizontal_sobel, filter_size)
    sobel_y = generic_function(padded_image.copy(), vertiacl_sobel, filter_size)
    magnitude = np.sqrt(sobel_x * 2 + sobel_y * 2)
    phase = np.arctan2(sobel_y, sobel_x)

    cv.imshow("horizontal_sobel_image", sobel_x)
    cv.imshow("Vertiacl_sobel_image", sobel_y)
    cv.waitKey(0)

    feature_index = 0
    for i in range(0, rows, cell_size):
        for j in range(0, cols, cell_size):
            cell = image[i:i + cell_size + 1, j:j + cell_size + 1]
            magnitude_cell = magnitude[i:i + cell_size + 1, j:j + cell_size + 1]
            phase_cell = phase[i:i + cell_size + 1, j:j + cell_size + 1]

            cell_rows, cell_cols = cell.shape
            histogram_bin = [0] * 6

            for k in range(cell_rows):
                for m in range(cell_cols):

                    phase_value = phase_cell[k, m]
                    magnitude_value = magnitude_cell[k, m]
                    if (0 <= phase_value <= 29):
                        histogram_bin[0] += magnitude_value
                    elif (30 <= phase_value <= 59):
                        histogram_bin[1] += magnitude_value
                    elif (60 <= phase_value <= 89):
                        histogram_bin[2] += magnitude_value
                    elif (90 <= phase_value <= 119):
                        histogram_bin[3] += magnitude_value
                    elif (120 <= phase_value <= 149):
                        histogram_bin[4] += magnitude_value
                    else:
                        histogram_bin[5] += magnitude_value

            feature_vector[feature_index:feature_index + 6] = histogram_bin
            feature_index += 6

    print("index stop ", feature_index)
    return feature_vector


image = cv.imread("D:/NUST/SEMESTER-6/DIP-LAB/DIP-VS CODE/LAB_10/image4.png", 0)
cv.imshow("Original Fig", image)
cv.waitKey(0)

feature_vector = hog_transform(image.copy())
print(feature_vector)

normalized_feature_vector = block_normalization(feature_vector)
print(normalized_feature_vector)