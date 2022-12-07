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
