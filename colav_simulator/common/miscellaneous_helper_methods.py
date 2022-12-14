"""
    miscellaneous_helper_methods.py

    Summary:
        Contains general utility functions.

    Author: Trym Tengesdal
"""

import math
from datetime import datetime
from typing import Tuple
from zoneinfo import ZoneInfo

import colav_simulator.common.math_functions as mf
import numpy as np
from scipy.stats import chi2


def create_probability_ellipse(P: np.ndarray, probability: float = 0.99) -> Tuple[list, list]:
    """Creates a probability ellipse for a covariance matrix P and a given
    confidence level (default 0.99).

    Args:
        P (np.ndarray): Covariance matrix
        probability (float, optional): Confidence level. Defaults to 0.99.

    Returns:
        np.ndarray: Ellipse data in x and y coordinates
    """

    # eigenvalues and eigenvectors of the covariance matrix
    eigenval, eigenvec = np.linalg.eig(P[0:2, 0:2])

    largest_eigenval = max(eigenval)
    largest_eigenvec_idx = np.argwhere(eigenval == max(eigenval))[0][0]
    largest_eigenvec = eigenvec[:, largest_eigenvec_idx]

    smallest_eigenval = min(eigenval)
    # if largest_eigenvec_idx == 0:
    #     smallest_eigenvec = eigenvec[:, 1]
    # else:
    #     smallest_eigenvec = eigenvec[:, 0]

    angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])
    angle = mf.wrap_angle_to_02pi(angle)

    # Get the ellipse scaling factor based on the confidence level
    chisquare_val = chi2.ppf(q=probability, df=2)

    a = chisquare_val * math.sqrt(largest_eigenval)
    b = chisquare_val * math.sqrt(smallest_eigenval)

    # the ellipse in "body" x and y coordinates
    t = np.linspace(0, 2.01 * np.pi, 100)
    x = a * np.cos(t)
    y = b * np.sin(t)

    R = mf.Rmtrx2D(angle)

    # Rotate to NED by angle phi, N_ell_points x 2
    ellipse_xy = np.array([x, y])
    for i in range(len(ellipse_xy)):
        ellipse_xy[:, i] = R @ ellipse_xy[:, i]

    return ellipse_xy[0, :].tolist(), ellipse_xy[1, :].tolist()


def get_list_except_element_idx(input_list: list, idx: int) -> list:
    """Returns a list with all elements of input_list except the element at idx.

    Args:
        input_list (list): List to get elements from
        idx (int): Index of element to exclude

    Returns:
        list: List with all elements of input_list except the element at idx
    """
    output_list = input_list.copy()
    output_list.pop(idx)
    return output_list


def get_relevant_do_states(input_list: list, idx: int) -> list:
    """Returns a tuple list of relevant dynamic obstacle indices, states to use in tracking/sensor generation
    , with all elements of input_list except the element <idx>, if this index is in the tuple list.

    Args:
        input_list (list): List of (do_idx, do_state) to get elements from
        idx (int): Index of element to exclude

    Returns:
        list: List with all (do_idx, do_state) tuples of input_list except the element idx, if idx is in the tuple list
    """
    output_list = []
    for do_idx, do_state in input_list:
        if do_idx != idx:
            output_list.append((do_idx, do_state))

    return output_list


def convert_sog_cog_state_to_vxvy_state(xs: np.ndarray) -> np.ndarray:
    """Converts from state(s) [x, y, U, chi] x N to [x, y, Vx, Vy] x N.

    Args:
        xs (np.ndarray): State(s) to convert.

    Returns:
        np.ndarray: Converted state.
    """

    if xs.ndim == 1:
        return np.array([xs[0], xs[1], xs[2] * np.cos(xs[3]), xs[2] * np.sin(xs[3])])
    else:
        return np.array(
            [xs[0, :], xs[1, :], np.multiply(xs[2, :], np.cos(xs[3, :])), np.multiply(xs[2, :], np.sin(xs[3, :]))]
        )


def convert_vxvy_state_to_sog_cog_state(xs: np.ndarray) -> np.ndarray:
    """Converts from a state [x, y, Vx, Vy] x N to [x, y, U, chi] x N.

    Args:
        xs (np.ndarray): State(s) to convert.

    Returns:
        np.ndarray: Converted state.
    """
    if xs.ndim == 1:
        return np.array([xs[0], xs[1], np.sqrt(xs[2] ** 2 + xs[3] ** 2), np.arctan2(xs[3], xs[2])])
    else:
        return np.array(
            [
                xs[0, :],
                xs[1, :],
                np.sqrt(np.multiply(xs[2, :], xs[2, :]) + np.multiply(xs[3, :], xs[3, :])),
                np.arctan2(xs[3, :], xs[2, :]),
            ]
        )


def current_utc_timestamp() -> int:
    """
    Returns:
        int: Current UTC timestamp
    """
    return int(datetime.utcnow().timestamp())


def utc_timestamp_to_local_time(timestamp: int) -> datetime:
    """
    Converts UTC timestamp to local time.

    Args:
        timestamp (int): UTC timestamp

    Returns:
        datetime: Local time
    """
    return utc_to_local(utc_timestamp_to_datetime(timestamp))


def utc_timestamp_to_datetime(timestamp: int) -> datetime:
    """
    Converts UTC timestamp to datetime.

    Args:
        timestamp (int): UTC timestamp

    Returns:
        datetime: Datetime object
    """
    return datetime.fromtimestamp(timestamp)


def local_timestamp_from_utc():
    """

    Returns:
        int: Current local time referenced timestamp
    """
    return datetime.now().astimezone().timestamp()


def utc_to_local(utc_dt) -> datetime:
    """
    Convert UTC datetime to local datetime.

    Parameters:
        utc_dt (datetime): UTC datetime

    Returns:
        datetime: Local datetime
    """
    return utc_dt.replace(tzinfo=ZoneInfo("localtime"))
