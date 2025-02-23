import numpy as np
import cv2


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
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read image")

    # Apply connected component labeling
    labels = connected_component(image)

    # Visualize results (normalize labels for display)
    normalized_labels = (255 * labels / labels.max()).astype(np.uint8)

    # Display results
    cv2.imshow("Original Image", image)
    cv2.imshow("Connected Components", normalized_labels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return labels

image = cv2.imread(r"C:\Users\sanan\Pictures\cc.png", cv2.IMREAD_GRAYSCALE)
segment_image(r"C:\Users\sanan\Pictures\cc.png")
