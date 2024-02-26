"""
    image_helper_methods.py

    Summary:
        Contains helper methods for image processing.

    Author: Trym Tengesdal
"""

import time
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sci_ndimage
import skimage.color as ski_color
import skimage.feature as ski_feature
import skimage.segmentation as ski_seg
import skimage.util as ski_util
from matplotlib import gridspec


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
    max_value: int = 255,
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


def create_simulation_image_segmentation_mask(img: np.ndarray) -> np.ndarray:
    """Thresholds the image from a simulation liveplot to create a mask for distinct features/edges in the image.

    Args:
        img (np.ndarray): The image to create a mask for.

    Returns:
        np.ndarray: The mask.
    """
    start_time = time.time()
    ndims = img.ndim
    if ndims == 3:
        img = np.expand_dims(img, axis=0)

    show_plots = True
    n_envs = img.shape[0]
    C, H, W = img.shape[1:]
    masks = np.zeros((n_envs, C, H, W), dtype=np.int16)
    for i in range(n_envs):
        for c in range(3):
            img_c = img[i, c]
            thresholds = [0.1, 0.3, 0.65]  # land, ships, nominal path
            img_c_scaled = img_c / 255

            binarized_gray_land = np.asarray(img_c_scaled < thresholds[0], dtype=np.uint8)
            binarized_gray_land_inv = np.asarray(ski_util.invert(binarized_gray_land), dtype=np.uint8)
            if show_plots and i == 0:
                fig = plt.figure(figsize=(5, 10))
                gs = gridspec.GridSpec(
                    1,
                    5,
                    fig,
                    wspace=0.0,
                    hspace=0.2,
                    top=0.95,
                    bottom=0.05,
                )
                ax = plt.subplot(gs[0])
                ax.imshow(img_c, cmap="gray")
                titles = ["Original", "Land mask", "Ship mask", "Nominal path mask", "Weighted mask"]
                ax.set_title(titles[0])
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax = plt.subplot(gs[1])
                ax.imshow(binarized_gray_land, cmap="gray")
                ax.set_title(titles[1])
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)

            mask_img_c = np.zeros_like(img_c, dtype=np.float16)
            mask_imgs = [binarized_gray_land.astype(np.float32)]
            for j in range(1, 3):
                binarized_gray = np.asarray(img_c_scaled > thresholds[j], dtype=np.uint8)
                if j == 1:
                    binarized_gray = np.asarray(ski_util.invert(binarized_gray), dtype=np.uint8)
                    binarized_gray = np.asarray(
                        ski_util.invert(binarized_gray * binarized_gray_land_inv), dtype=np.uint8
                    )
                else:
                    ship_mask_inv = np.asarray(ski_util.invert(mask_imgs[1]), dtype=np.uint8)
                    binarized_gray = binarized_gray * ship_mask_inv

                median = cv2.medianBlur(binarized_gray, 5).astype(np.int16)
                median = (median - np.min(median)) / (np.max(median) - np.min(median) + 1e-6)
                mask_imgs.append(median)

                if show_plots and i == 0:
                    ax = plt.subplot(gs[j + 1])
                    ax.set_title(titles[j + 1])
                    ax.imshow(median, cmap="gray")
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)

            mask_img_c = (
                1.0 * mask_imgs[0].astype(np.int16)
                + 500.0 * mask_imgs[1].astype(np.int16)
                + 100.0 * mask_imgs[2].astype(np.int16)
            )
            masks[i, c] = mask_img_c

            if show_plots and i == 0:
                ax = plt.subplot(gs[-1])
                ax.imshow(mask_img_c)
                ax.set_title("Weighted mask")
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                plt.subplots_adjust(wspace=0.01, hspace=0.01)
                plt.tight_layout()

                # plt.savefig("segmentation.pdf", format="pdf")

            # segments_fz = ski_seg.felzenszwalb(img_c, scale=100, sigma=0.5, min_size=50)
            # segments_slic = ski_seg.slic(
            #     img_c, n_segments=250, compactness=0.1, sigma=1, start_label=1, channel_axis=None
            # )
            # segments_quick = ski_seg.quickshift(img_c, kernel_size=3, max_dist=6, ratio=0.5)
            # gradient = ski_seg.sobel(img_c)
            # segments_watershed = ski_seg.watershed(gradient, markers=250, compactness=0.001)

            # _, binary_threshold = cv2.threshold(img_c, 150, 170, cv2.THRESH_BINARY)
            # _, otsu_threshold = cv2.threshold(img_c, 0, 255, cv2.THRESH_OTSU)
            # adaptive_gaussian_threshold = cv2.adaptiveThreshold(
            #     img_c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            # )
            # contours, hierarchy = cv2.findContours(adaptive_gaussian_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # contoured_img = adaptive_gaussian_threshold.copy()
            # for cnt in contours:
            #     cv2.drawContours(contoured_img, [cnt], 0, 255, -1)

            # D = sci_ndimage.distance_transform_edt(contoured_img)
            # peak_coords = ski_feature.peak_local_max(D, min_distance=5, labels=adaptive_gaussian_threshold)
            # local_max_mask = np.zeros(D.shape, dtype=bool)
            # local_max_mask[tuple(peak_coords.T)] = True
            # markers, _ = sci_ndimage.label(local_max_mask)
            # labels = ski_seg.watershed(-D, markers, mask=contoured_img)
            # print(f"[INFO] {len(np.unique(labels)) - 1} unique segments found")

    print(f"[INFO] Time to create segmentation mask: {time.time() - start_time:.2f} seconds")
    return masks


if __name__ == "__main__":

    # filename_in = "input.png"
    # filename_out = "output.png"

    # image = cv2.imread(str(filename_in))
    # img_cropped = remove_whitespace(image)
    # plt.imshow(img_cropped)
    # cv2.imwrite(str(filename_out), img_cropped)

    data_dir = Path("/home/doctor/Desktop/machine_learning/data/vae/")
    # data_dir = Path("/Users/trtengesdal/Desktop/machine_learning/data/vae/")
    npy_filename = "perception_images_rogaland_random_everything_vecenv_test"

    npy_file = np.load(data_dir / (npy_filename + ".npy"), mmap_mode="r", allow_pickle=True).astype(np.uint8)

    segmask = create_simulation_image_segmentation_mask(npy_file[0])
