"""Contains commonly used math functions and utility functions."""
import math
import os
import shutil
from datetime import datetime, timedelta

import numpy as np
import pytz


def wrap_angle_to_pmpi(angle):
    """Wraps input angle to [-pi, pi)

    Args:
        angle (float): Angle in radians

    Returns:
        float: Wrapped angle
    """
    while angle <= -math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle

def wrap_angle_to_02pi(angle):
    """Wraps input angle to [0, 2pi)

    Args:
        angle (float): Angle in radians

    Returns:
        float: Wrapped angle
    """
    while angle <= 0:
        angle += 2 * math.pi
    while angle > 2 * math.pi:
        angle -= 2 * math.pi
    return angle

def wrap_angle_diff_to_pmpi(a_1, a_2):
    """Wraps angle difference a_1 - a_2 to within [-pi, pi)

    Args:
        a_1 (float): Angle in radians
        a_2 (float): Angle in radians

    Returns:
        _type_: Wrapped angle difference
    """
    diff = wrap_angle_to_pmpi(a_1) - wrap_angle_to_pmpi(a_2)
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff

def knots2mps(knots):
    """Converts from knots to miles per second.

    Args:
        knots (float): Knots to convert.

    Returns:
        float: Knots converted to mps.
    """
    mps = knots * 1.852 / 3.6
    return mps


def mps2knots(mps):
    """Converts from miles per second to knots.

    Args:
        mps (float): Mps to convert.

    Returns:
        float: mps converted to knots.
    """
    knots = mps * 3.6 / 1.852
    return knots


def seconds_to_date_time_utc(time: int):
    """
		Converts seconds to date time UTC format
		Time=0 -> 2021-01-01 00:00:00+00:00
		Note: time must be int, float will produce error
	"""
    hms_time = str(timedelta(seconds=time))
    local = pytz.timezone("Europe/London")  # London to get UTC+0
    naive = datetime.strptime(f"2021-1-1 {hms_time}", "%Y-%m-%d %H:%M:%S")
    local_dt = local.localize(naive, is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc)
    utc_dt.strftime("%Y-%m-%d %H:%M:%S")
    return utc_dt


def normalize_vec(v: np.ndarray):
    """Normalize vector v to length 1.

    Args:
        v (np.ndarray): Vector to normalize

    Returns:
        np.ndarray: Normalized vector
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    else:
        return v / norm

# necessary??
def move_xlsx_files():
    for file in os.listdir():
        if file.endswith('.xlsx'):
            if len(file) == 17:
                shutil.move(file, 'output/eval/ship' + str(file[11]) + '/' + file)
            if len(file) == 18:
                shutil.move(file, 'output/eval/ship' + str(file[11]) + str(file[12]) + '/' + file)
            if len(file) == 19:
                shutil.move(file, 'output/eval/ship' + str(file[11]) + str(file[12]) + str(file[13]) + '/' + file)
