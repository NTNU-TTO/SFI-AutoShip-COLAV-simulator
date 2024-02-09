"""
    image_helper_methods.py

    Summary:
        Contains helper methods for image processing.

    Author: Trym Tengesdal
"""

from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def find_edges(img: np.ndarray, *, bw_threshold: int = 150, limits: Tuple[float, float] = (0.2, 0.15)) -> list:
    mask = img < bw_threshold
    edges = []
    for axis in (1, 0):
        count = mask.sum(axis=axis)
        limit = limits[axis] * img.shape[axis]
        index_ = np.where(count >= limit)
        _min, _max = index_[0][0], index_[0][-1]
        edges.append((_min, _max))
    return edges


def preprocess_image(
    img: np.ndarray,
    *,
    kernel_size: int = 15,
    crop_side: int = 50,
    block_size: int = 35,
    constant: int = 15,
    max_value: int = 255
) -> np.ndarray:
    """Preprocess the image before further processing.

    Args:
        img (np.ndarray): The image to preprocess.
        kernel_size (int, optional): Size of the kernel for erosion. Defaults to 15.
        crop_side (int, optional): Number of pixels to crop from each side. Defaults to 50.
        block_size (int, optional): Size of the block for adaptive thresholding. Defaults to 35.
        constant (int, optional): Constant for adaptive thresholding. Defaults to 15.
        max_value (int, optional): Maximum value for adaptive thresholding. Defaults to 255.

    Returns:
        np.ndarray: Preprocessed image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bit = cv2.bitwise_not(gray)
    image_adapted = cv2.adaptiveThreshold(
        src=bit,
        maxValue=max_value,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=constant,
    )
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(image_adapted, kernel, iterations=2)
    return erosion[crop_side:-crop_side, crop_side:-crop_side]


def adapt_edges(edges: list, *, height: int, width: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Adapt the edges to fit the image.

    Args:
        edges (list): List of edges.
        height (int): Image height.
        width (int): Image width.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: Adapted edges.
    """
    (x_min, x_max), (y_min, y_max) = edges
    x_min2 = x_min
    x_max2 = x_max + min(250, (height - x_max) * 10 // 11)
    # could do with less magic numbers
    y_min2 = max(0, y_min)
    y_max2 = y_max + min(250, (width - y_max) * 10 // 11)
    return (x_min2, x_max2), (y_min2, y_max2)


def remove_whitespace(img: np.ndarray) -> np.ndarray:
    """Remove whitespace from around the image.

    Args:
        img (np.ndarray): The image to remove whitespace from.

    Returns:
        np.ndarray: Cropped image.
    """
    height, width = img.shape[0:2]
    img_preprocessed = preprocess_image(img)
    edges = find_edges(img_preprocessed)
    (x_min, x_max), (y_min, y_max) = adapt_edges(edges, height=height, width=width)
    image_cropped = img[x_min:x_max, y_min:y_max]
    return image_cropped


if __name__ == "__main__":

    filename_in = "input.png"
    filename_out = "output.png"

    image = cv2.imread(str(filename_in))
    img_cropped = remove_whitespace(image)
    plt.imshow(img_cropped)
    cv2.imwrite(str(filename_out), img_cropped)
