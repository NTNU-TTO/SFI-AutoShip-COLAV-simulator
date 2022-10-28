"""Contains general non-math related utility functions."""

import os
import shutil
from datetime import datetime, timedelta

import pytz


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
