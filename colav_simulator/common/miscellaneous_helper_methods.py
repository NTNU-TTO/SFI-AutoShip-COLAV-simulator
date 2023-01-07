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

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import numpy as np
import pandas as pd
from colav_evaluation_tool.vessel import VesselData, compute_total_dist_travelled
from scipy.stats import chi2


def convert_simulation_data_to_vessel_data(sim_data: pd.DataFrame, ship_info: dict, utm_zone: int) -> list:
    """Converts simulation data to vessel data.

    Args:
        sim_data (pd.DataFrame): Simulation data.
        ship_info (dict): Information on all ships in the simulation.
        utm_zone (int): UTM zone used for the planar xy coordinate system.

    Returns:
        list: List of vessel data.
    """
    vessels = []
    identifier = 0
    for name, ship_i_info in ship_info.items():
        vessel = VesselData(
            id=identifier,
            name=name,
            mmsi=ship_i_info["mmsi"],
            length=ship_i_info["length"],
            width=ship_i_info["width"],
            draft=ship_i_info["draft"],
        )

        X, vessel.timestamps, vessel.datetimes_utc = extract_trajectory_data_from_ship_dataframe(sim_data[name])

        vessel.first_valid_idx, vessel.last_valid_idx = index_of_first_and_last_non_nan(X[0, :])
        vessel.n_msgs = len(vessel.timestamps)
        vessel.xy = np.zeros((2, len(X[0, :])))
        vessel.xy[0, :] = X[1, :]  # ENU frame is used in the evaluator
        vessel.xy[1, :] = X[0, :]
        vessel.sog = X[2, :]
        vessel.cog = X[3, :]

        vessel.latlon = np.zeros(vessel.xy.shape) * np.nan
        (vessel.latlon[0, vessel.first_valid_idx : vessel.last_valid_idx + 1], vessel.latlon[1, vessel.first_valid_idx : vessel.last_valid_idx + 1],) = mapf.local2latlon(
            vessel.xy[0, vessel.first_valid_idx : vessel.last_valid_idx + 1],
            vessel.xy[1, vessel.first_valid_idx : vessel.last_valid_idx + 1],
            utm_zone,
        )

        vessel.travel_dist = compute_total_dist_travelled(vessel.xy[:, vessel.first_valid_idx : vessel.last_valid_idx + 1])

        print(f"Vessel {identifier} travelled a distance of {vessel.travel_dist} m")
        print(f"Vessel status: {vessel.status}")

        vessels.append(vessel)

        identifier += 1

    return vessels


def extract_trajectory_data_from_ship_dataframe(ship_df: pd.DataFrame) -> Tuple[np.ndarray, list, list]:
    """Extract the trajectory related data from a ship dataframe.

    Args:
        ship_df (Dataframe): Dataframe containing the ship simulation data.

    Returns:
        Tuple[np.ndarray, list, list]: Tuple of array containing the trajectory and corresponding relative simulation timestamps and UTC timestamps.
    """
    X = np.zeros((4, len(ship_df)))
    timestamps = []
    datetimes_utc = []
    for k, ship_df_k in enumerate(ship_df):
        X[:, k] = ship_df_k["pose"]
        timestamps.append(float(ship_df_k["timestamp"]))
        datetime_utc = datetime.strptime(ship_df_k["date_time_utc"], "%d.%m.%Y %H:%M:%S")
        datetimes_utc.append(datetime_utc)

    return X, timestamps, datetimes_utc


def extract_track_data_from_dataframe(ship_df: pd.DataFrame) -> dict:
    """Extract the dynamic obstacle track data from a ship dataframe.

    Args:
        ship_df (Dataframe): Dataframe containing the ship simulation data.

    Returns:
        Tuple[list, list]: List of dynamic obstacle estimates and covariances
    """
    output = {}
    do_estimates = []
    do_covariances = []
    do_NISes = []

    n_samples = len(ship_df)
    n_do = len(ship_df[n_samples - 1]["do_estimates"])
    do_labels = ship_df[n_samples - 1]["do_labels"]

    for i in range(n_do):
        do_estimates.append(np.nan * np.ones((4, n_samples)))
        do_covariances.append(np.nan * np.ones((4, 4, n_samples)))
        do_NISes.append(np.nan * np.ones(n_samples))

    for i in range(n_do):
        for k, ship_df_k in enumerate(ship_df):
            for idx, _ in enumerate(ship_df_k["do_labels"]):
                do_estimates[idx][:, k] = ship_df_k["do_estimates"][idx]
                do_covariances[idx][:, :, k] = ship_df_k["do_covariances"][idx]
                do_NISes[idx][k] = ship_df_k["do_NISes"][idx]

    output["do_estimates"] = do_estimates
    output["do_covariances"] = do_covariances
    output["do_NISes"] = do_NISes
    output["do_labels"] = do_labels
    return output


def check_if_trajectory_is_within_xy_limits(trajectory: np.ndarray, xlimits: list, ylimits: list) -> bool:
    """Checks if the trajectory is within the x and y limits.

    Args:
        trajectory (np.ndarray): Trajectory data.
        xlimits (list): List containing the x limits.
        ylimits (list): List containing the y limits.

    Returns:
        bool: True if trajectory is within limits, False otherwise.
    """
    min_x = np.min(trajectory[0, :])
    max_x = np.max(trajectory[0, :])
    min_y = np.min(trajectory[1, :])
    max_y = np.max(trajectory[1, :])
    return min_x >= xlimits[0] and max_x <= xlimits[1] and min_y >= ylimits[0] and max_y <= ylimits[1]


def update_xy_limits_from_trajectory_data(trajectory: np.ndarray, xlimits: list, ylimits: list) -> Tuple[list, list]:
    """Update the x and y limits from the trajectory data (either predefined trajectory or nominal trajectory/waypoints for the ship).

    Args:
        X (np.ndarray): waypoint data.
        xlimits (list): List containing the x limits.
        ylimits (list): List containing the y limits.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y limits.
    """

    min_x = np.min(trajectory[0, :])
    max_x = np.max(trajectory[0, :])
    min_y = np.min(trajectory[1, :])
    max_y = np.max(trajectory[1, :])

    if min_x < xlimits[0]:
        xlimits[0] = min_x

    if max_x > xlimits[1]:
        xlimits[1] = max_x

    if min_y < ylimits[0]:
        ylimits[0] = min_y

    if max_y > ylimits[1]:
        ylimits[1] = max_y

    return xlimits, ylimits


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
        return np.array([xs[0, :], xs[1, :], np.multiply(xs[2, :], np.cos(xs[3, :])), np.multiply(xs[2, :], np.sin(xs[3, :]))])


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


def index_of_first_and_last_non_nan(input_list: list | np.ndarray) -> Tuple[int, int]:
    """Returns the index of the first and last non-NaN element in a list or numpy array."""

    if isinstance(input_list, list):
        input_list = np.array(input_list)

    non_nan_indices = np.where(~np.isnan(input_list))[0]
    if non_nan_indices.size > 0:
        first_non_nan_idx = int(non_nan_indices[0])
        last_non_nan_idx = int(non_nan_indices[-1])
    else:
        first_non_nan_idx = -1
        last_non_nan_idx = -1

    return first_non_nan_idx, last_non_nan_idx


def current_utc_datetime_str(format_str: str) -> str:
    """Returns the current date and time as a string with specified format.

    Args:
        format (str): Format of the datetime string, e.g. %Y%m%d_%H_%M_%S
    """
    return datetime.utcnow().strftime(format_str)


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


def local_timestamp_from_utc() -> int:
    """

    Returns:
        int: Current local time referenced timestamp
    """
    return datetime.now().astimezone().timestamp()


def utc_to_local(utc_dt: datetime) -> datetime:
    """
    Convert UTC datetime to local datetime.

    Parameters:
        utc_dt (datetime): UTC datetime

    Returns:
        datetime: Local datetime
    """
    return utc_dt.replace(tzinfo=ZoneInfo("localtime"))
