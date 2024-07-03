"""
    image_helper_methods.py

    Summary:
        Contains helper methods for image processing.

    Author: Trym Tengesdal
"""

import time
from pathlib import Path
from typing import Tuple

import colav_simulator.common.math_functions as mf
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

matplotlib.use("Agg")
import numpy as np
import skimage.morphology as ski_morph
import skimage.util as ski_util
from matplotlib import gridspec

# Depending on your OS, you might need to change these paths
plt.rcParams["animation.convert_path"] = "/usr/bin/convert"
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"


def save_frames_as_gif(frame_list: list, filename: Path, verbose: bool = False) -> None:
    # Mess with this to change frame size
    fig = plt.figure(figsize=(frame_list[0].shape[1] / 72.0, frame_list[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frame_list[0], aspect="auto")
    plt.axis("off")

    def init():
        patch.set_data(frame_list[0])
        return (patch,)

    def animate(i):
        patch.set_data(frame_list[i])
        return (patch,)

    anim = animation.FuncAnimation(
        fig=fig, func=animate, init_func=init, blit=True, frames=len(frame_list), interval=50, repeat=True
    )
    anim.save(
        filename=filename.as_posix(),
        writer=animation.PillowWriter(fps=20),
        progress_callback=lambda i, n: print(f"Saving frame {i} of {n}") if verbose else None,
    )
    print(f"Saved gif to {filename}")


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
    """Preprocess an image for whitespace removal.

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


def gray_to_rgb(img):
    return np.repeat(img, 3, 2)


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

    show_plots = False
    n_envs = img.shape[0]
    C, H, W = img.shape[1:]
    masks = np.zeros((n_envs, C, H, W), dtype=np.int16)
    for i in range(n_envs):
        for c in range(C):
            img_c = img[i, c]
            thresholds = [
                [0.1, 0.6],
                [0.05, 0.57],
                [0.63, 1.1],
            ]  # land, ownship, obstacle ships, nominal path, [lower, upper]
            img_c_scaled = img_c / 255

            if show_plots and i == 0 and c == 0:
                fig = plt.figure(figsize=(10, 5))
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

            mask_img_c = np.zeros_like(img_c, dtype=np.float16)
            mask_imgs = []
            for j in range(3):
                threshold_j = np.asarray(img_c_scaled > thresholds[j][0], dtype=np.float16)
                if j > 0:
                    threshold_j = threshold_j * np.asarray(img_c_scaled < thresholds[j][1], dtype=np.float16)
                    if j == 1:
                        land_mask_inv = np.asarray(ski_util.invert(mask_imgs[0]), dtype=np.float16)
                        threshold_j = np.asarray(ski_util.invert(threshold_j * land_mask_inv), dtype=np.uint8)
                    elif j == 2:
                        ship_mask_inv = np.asarray(ski_util.invert(mask_imgs[1]), dtype=np.float16)
                        threshold_j = (threshold_j * ship_mask_inv).astype(np.uint8)
                    # obstacles

                else:
                    threshold_j = np.asarray(ski_util.invert(threshold_j), dtype=np.uint8)

                mask_level_j = threshold_j
                if j == 1:
                    # mask_level_j = cv2.bilateralFilter(threshold_j, 8, 75, 75).astype(np.float16)
                    mask_level_j = ski_morph.erosion(mask_level_j.astype(np.uint8), ski_morph.disk(2)).astype(
                        np.float16
                    )
                    mask_level_j = ski_morph.dilation(mask_level_j.astype(np.uint8), ski_morph.disk(2)).astype(
                        np.float16
                    )
                elif j == 2:
                    mask_level_j = cv2.medianBlur(threshold_j, 5).astype(np.float16)

                if mask_level_j.max() > 1.0:
                    mask_level_j = mf.linear_map(mask_level_j, (mask_level_j.min(), mask_level_j.max()), (0.0, 1.0))

                mask_imgs.append(mask_level_j)

                if show_plots and i == 0 and c == 0:
                    ax = plt.subplot(gs[j + 1])
                    ax.set_title(titles[j + 1])
                    ax.imshow(mask_level_j.astype(np.uint8), cmap="gray")
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)

            contours, hierarchy = cv2.findContours(
                mask_imgs[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            land_mask_edges = (
                cv2.drawContours(np.zeros_like(mask_imgs[0], dtype=np.uint8), contours, -1, 255, 1).astype(np.int16)
                / 255
            )
            # ax.imshow(land_mask_edges.astype(np.uint8), cmap="gray")

            mask_img_c = (
                25.0 * mask_imgs[0].astype(np.int16)  # land
                + 100.0 * land_mask_edges  # land edges
                + 250.0 * mask_imgs[1].astype(np.int16)
                + 150.0 * mask_imgs[2].astype(np.int16)
            )
            mask_img_c[mask_img_c < 1.0] = 1.0

            if show_plots and i == 0 and c == 0:
                ax = plt.subplot(gs[-1])
                ax.imshow(mask_img_c.astype(np.uint8), cmap="gray")
                ax.set_title("Weighted mask")
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)

            masks[i, c] = mask_img_c

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
