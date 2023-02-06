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


def check_if_vessel_is_passed_by(
    p_os: np.ndarray, v_os: np.ndarray, p_do: np.ndarray, v_do: np.ndarray, threshold_angle: float = 100.0, threshold_distance: float = 50.0
) -> bool:
    """Checks if a vessel is passed by another vessel.

    Args:
        p_os (_type_): Position of ownship
        v_os (_type_): Velocity of ownship
        p_do (_type_): Position of dynamic obstacle
        v_do (_type_): Velocity of dynamic obstacle

    Returns:
        bool: True if ownship is passed by dynamic obstacle, False otherwise.
    """
    dist_os_do = p_do - p_os
    L_os_do = dist_os_do / np.linalg.norm(dist_os_do)
    U_os = np.linalg.norm(v_os)
    U_do = np.linalg.norm(v_do)

    os_is_overtaken = np.dot(v_os, v_do) > np.cos(np.deg2rad(68.5)) * U_os * U_do and U_os < U_do and U_os > 0.25
    do_is_overtaken = np.dot(v_do, v_os) > np.cos(np.deg2rad(68.5)) * U_do * U_os and U_do < U_os and U_do > 0.25

    vessel_is_passed: bool = (
        (np.dot(v_os, L_os_do) < np.cos(np.deg2rad(threshold_angle)) * U_os and not os_is_overtaken)
        or (np.dot(v_do, -L_os_do) < np.cos(np.deg2rad(threshold_angle)) * U_do and not do_is_overtaken)
        and np.linalg.norm(dist_os_do) > threshold_distance
    )

    return vessel_is_passed


def compute_vessel_pair_cpa(p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Computes the closest point of approach (CPA) between two vessel when assumed to travel with constant velocity.

    Args:
        p1 (np.ndarray): Position of vessel 1.
        v1 (np.ndarray): Velocity of vessel 1.
        p2 (np.ndarray): Position of vessel 2.
        v2 (np.ndarray): Velocity of vessel 2.

    Returns:
        Tuple[float, float, np.ndarray]: The time to CPA, distance at CPA and corresponding CPA distance vector.
    """
    # Compute the relative position and velocity
    r = p2 - p1
    v = v2 - v1

    if np.dot(v, v) < 0.0001:
        return 0.0, float(np.linalg.norm(r)), p2 - p1

    t_cpa = -np.dot(r, v) / np.dot(v, v)
    d_cpa_vec = r + t_cpa * v
    d_cpa = float(np.linalg.norm(d_cpa_vec))
    return t_cpa, d_cpa, d_cpa_vec


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
        n_msgs = len(vessel.timestamps)
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

        vessel.forward_heading_estimate = np.zeros(n_msgs) * np.nan
        vessel.backward_heading_estimate = np.zeros(n_msgs) * np.nan
        for k in range(vessel.first_valid_idx, vessel.last_valid_idx):
            vessel.forward_heading_estimate[k] = np.arctan2(vessel.xy[1, k + 1] - vessel.xy[1, k], vessel.xy[0, k + 1] - vessel.xy[0, k])
        vessel.forward_heading_estimate[vessel.last_valid_idx] = vessel.forward_heading_estimate[vessel.last_valid_idx - 1]

        for k in range(vessel.first_valid_idx + 1, vessel.last_valid_idx):
            vessel.backward_heading_estimate[k] = np.arctan2(vessel.xy[1, k] - vessel.xy[1, k - 1], vessel.xy[0, k] - vessel.xy[0, k - 1])
        vessel.backward_heading_estimate[vessel.first_valid_idx] = vessel.forward_heading_estimate[vessel.first_valid_idx]

        vessel.travel_dist = compute_total_dist_travelled(vessel.xy[:, vessel.first_valid_idx : vessel.last_valid_idx + 1])

        print(f"Vessel {identifier} travelled a distance of {vessel.travel_dist} m")
        print(f"Vessel status: {vessel.status}")

        vessels.append(vessel)

        identifier += 1

    return vessels


def find_cpa_indices(trajectory_list: list) -> np.ndarray:
    """Find the indices of the ships that are closest to each other.

    Args:
        trajectory_list (list): List of trajectories.

    Returns:
        np.ndarray: Array containing the indices of the ships that are closest to each other.
    """
    n_ships = len(trajectory_list)
    cpa_indices = np.zeros((n_ships, n_ships), dtype=int) * np.nan

    for i in range(n_ships):
        for j in range(n_ships):
            if i == j:
                continue
            cpa_indices[i, j] = find_cpa_index(trajectory_list[i], trajectory_list[j])

    return cpa_indices


def find_cpa_index(trajectory_i, trajectory_j) -> int | float:
    """Find the index of the closest point of approach between two ships.

    Args:
        trajectory_i (np.ndarray): Trajectory of ship i.
        trajectory_j (np.ndarray): Trajectory of ship j.

    Returns:
        int: Index of the closest point of approach for the trajectory pair.
    """
    min_dist_idx = np.nan

    n_samples = len(trajectory_i[0, :])
    ranges = np.zeros(n_samples)
    for k in range(len(trajectory_i[0, :])):
        ranges[k] = np.linalg.norm(trajectory_i[0:2, k] - trajectory_j[0:2, k])

    finite = np.where(~np.isnan(ranges))[0]
    if finite.size > 0:
        min_dist_idx = np.argmin(ranges[finite])
    return min_dist_idx


def extract_ship_data_from_sim_dataframe(ship_list: list, sim_data: pd.DataFrame) -> dict:
    """Extract the ship data from the simulation data.

    Args:
        ship_list (list): List of ship objects.
        sim_data (pd.DataFrame): Simulation data from the ships.
    Returns:
        dict: Dictionary containing the ship data related to the simulation.
    """
    output = {}
    trajectory_list = []
    for i, ship in enumerate(ship_list):
        X, timestamps, _ = extract_trajectory_data_from_ship_dataframe(sim_data[f"Ship{i}"])
        trajectory_list.append(X)

    output["trajectory_list"] = trajectory_list
    output["timestamps"] = timestamps
    cpa_indices = find_cpa_indices(trajectory_list)
    output["cpa_indices"] = cpa_indices
    return output


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
        X[:, k] = ship_df_k["csog_state"]
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


def convert_csog_state_to_vxvy_state(xs: np.ndarray) -> np.ndarray:
    """Converts from state(s) [x, y, U, chi] x N to [x, y, Vx, Vy] x N,
    where U is the speed over ground and chi is the course over ground.

    Args:
        xs (np.ndarray): State(s) to convert.

    Returns:
        np.ndarray: Converted state(s).
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
