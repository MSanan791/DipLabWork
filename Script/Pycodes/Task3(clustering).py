import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def kmeans_segmentation(image_path, k=3):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    mean_intensity = np.mean(img)
    k_values = np.linspace(mean_intensity - 40, mean_intensity + 40, k)

    segmented_img = np.copy(img)

    while True:
        distances = [np.abs(img - kv) for kv in k_values]
        labels = np.argmin(distances, axis=0)

        new_k_values = []
        for i in range(k):
            cluster = img[labels == i]
            new_k_values.append(np.mean(cluster) if len(cluster) > 0 else k_values[i])

        if np.allclose(new_k_values, k_values):
            break

        k_values = new_k_values

    for i in range(k):
        segmented_img[labels == i] = 255 - (255 // (k - 1)) * i

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_img, cmap='gray')
    plt.title(f"Segmented Image (K-Means, k={k})")
    plt.axis("off")

    plt.show()

# Example usage
image_path = r"Threshold_Image.png"
kmeans_segmentation(image_path, k=3)