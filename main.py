import numpy as np
import os
import cv2
import re
import math
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure


#################################HogDescriptor######################################

def compute_gradients(image):
    # Filter for computing gradients
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Gradient operations
    grad_x = np.zeros(image.shape)
    grad_y = np.zeros(image.shape)

    # Avoid boundaries for simplicity
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            grad_x[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * gx)
            grad_y[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * gy)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 180
    
    return magnitude, orientation, grad_x, grad_y

def cell_histograms(magnitude, orientation, cell_size=8, bin_size=20):
    # Initialize histograms
    max_angle = 180
    bins = np.arange(0, max_angle + bin_size, bin_size)
    cells_per_row = magnitude.shape[1] // cell_size
    cells_per_col = magnitude.shape[0] // cell_size
    histograms = np.zeros((cells_per_col, cells_per_row, len(bins) - 1))

    for i in range(cells_per_col):
        for j in range(cells_per_row):
            block = orientation[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            mag = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            hist, _ = np.histogram(block, bins=bins, weights=mag, density=True)
            histograms[i, j, :] = hist
    return histograms

def block_normalization(histograms, block_size=2):
    # Combine histograms in a larger block and normalize
    e = 1e-5  # To avoid division by zero
    blocks_per_row = histograms.shape[1] - block_size + 1
    blocks_per_col = histograms.shape[0] - block_size + 1
    normalized_blocks = np.zeros((blocks_per_col, blocks_per_row, histograms.shape[2] * block_size**2))

    for y in range(blocks_per_col):
        for x in range(blocks_per_row):
            block = histograms[y:y+block_size, x:x+block_size, :]
            norm = np.sqrt(np.sum(block**2) + e**2)
            normalized_blocks[y, x, :] = block.flatten() / norm
    return normalized_blocks.flatten()

def compute_hog_Descriptor(image, cell_size=8, block_size=2, bin_size=20):
    # Convert to grayscale if image is RGB
    if len(image.shape) > 2 and image.shape[2] == 3:
         image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    
    magnitude, orientation, *_ = compute_gradients(image)
    histograms = cell_histograms(magnitude, orientation, cell_size, bin_size)
    hog_descriptor = block_normalization(histograms, block_size)
    
    return hog_descriptor

def compute_distance(descriptor1, descriptor2):
    # Step 1: Compute the difference
    diff = descriptor1 - descriptor2  
    # Step 2: Square the differences
    squared_diff = np.square(diff)    
    # Step 3: Sum all the squared differences
    sum_of_squares = np.sum(squared_diff)    
    # Step 4: Take the square root to get the Euclidean distance
    distance = np.sqrt(sum_of_squares)
    
    return distance

def match_hog_descriptors(test_descriptor, template_descriptors):
    matches = []
    for template_descriptor in template_descriptors:
        if test_descriptor.shape != template_descriptor.shape:
            print(f"Shape mismatch: Test {test_descriptor.shape}, Template {template_descriptor.shape}")
            continue  # Skip this template or handle mismatch appropriately
        # Calculate the Euclidean distance between the HOG descriptors
        distance = compute_distance(test_descriptor,template_descriptor)
        matches.append(distance)
    return matches

def preprocess_image_with_hog(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (64, 64))  # Resize to fit HOG descriptor window size
    # Calculate HOG descriptors
    global hog_descriptors
    hog_descriptors = compute_hog_Descriptor(img_resized)
    return hog_descriptors




##########################################################################
def show_hog_features(image_path, ax, title):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(image, (64, 64))  # Resize to ensure the HOG features are visible

    # Compute HOG features and also get a visual representation
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # Setup the plots
    ax[0].imshow(resized_img, cmap='gray')
    ax[0].set_title(title)
    ax[0].axis('off')

    ax[1].imshow(hog_image_rescaled, cmap='gray')
    ax[1].set_title(title + ' - HOG Features')
    ax[1].axis('off')


cv2.waitKey(0)
cv2.destroyAllWindows()