import math
import numpy as np
from datetime import datetime, timedelta
import pytz
from ship_models import telemetron, random_ship_model
import shutil
import os


def create_ship_model(ship_model_name='random'):
    if ship_model_name == 'telemetron':
        return telemetron.Telemetron()
    else:
        return random_ship_model.Random_ship_model()


def wrap_to_pi(angle):
    # wraps the angle to [0,2*pi)
    res = math.fmod(angle + 2 * math.pi, 2 * math.pi)
    # if res > math.pi: res -= 2*math.pi #wraps the angle to [-pi,pi)
    return res


def normalize_angle(angle):
    # wraps the angle to [-pi, pi)
    while angle <= -math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi

    return angle


def normalize_angle_diff(angle, angle_ref):
    diff = angle_ref - angle

    if (diff > 0):
        new_angle = angle + (diff - math.fmod(diff, 2 * math.pi))
    else:
        new_angle = angle + (diff + math.fmod(-diff, 2 * math.pi))

    diff = angle_ref - new_angle
    if (diff > math.pi):
        new_angle += 2 * math.pi
    elif (diff < -math.pi):
        new_angle -= 2 * math.pi
    return new_angle


def knots2mps(knots):
    mps = knots * 1.852 / 3.6
    return mps


def mps2knots(mps):
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
    """
		Return normalized vector
	"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    else:
        return v / norm


""""def lat_long_dist_to_metres(lon1, lat1, lon2, lat2):
    if lon1 < 0 or lat1 < 0 or lon2 < 0 or lat2 < 0:
        return 999999
    r = 6362.132
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)
    d = r * c * 1000
    return round(d, 1)"""


def move_xlsx_files():
    for file in os.listdir():
        if file.endswith('.xlsx'):
            if len(file) == 17:
                shutil.move(file, 'output/eval/ship' + str(file[11]) + '/' + file)
            if len(file) == 18:
                shutil.move(file, 'output/eval/ship' + str(file[11]) + str(file[12]) + '/' + file)
            if len(file) == 19:
                shutil.move(file, 'output/eval/ship' + str(file[11]) + str(file[12]) + str(file[13]) + '/' + file)

