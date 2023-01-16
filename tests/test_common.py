from datetime import datetime
from zoneinfo import ZoneInfo

import colav_simulator.common.map_functions as map_functions
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
from seacharts.enc import ENC
from shapely.geometry import Point


def test_utc_timestamp_to_local_time():
    """Test UTC timestamp to local time."""
    timestamp = 1615450500
    local_time = mhm.utc_timestamp_to_local_time(timestamp)
    ground_truth = datetime(2021, 3, 11, 9, 15, 0, tzinfo=ZoneInfo("localtime"))
    assert local_time == ground_truth


def test_wrap_angle_to_pmpi():
    """Test wrap angle to [-pi, pi)."""
    angle = 0
    wrapped_angle = mf.wrap_angle_to_pmpi(angle)
    assert wrapped_angle == 0


def test_randomize_start_position():
    """Test randomize start position functionality."""
    enc = ENC(new_data=False)

    x0, y0 = map_functions.generate_random_start_position_from_draft(enc, 4.0)
    assert enc.seabed[5].geometry.contains(Point(y0, x0))


def test_min_distance_to_land():
    """Test min distance to land functionality."""
    enc = ENC(new_data=False)

    x0, y0 = map_functions.generate_random_start_position_from_draft(enc, 4.0)
    min_dist = map_functions.min_distance_to_land(enc, y0, x0)
    assert min_dist < 4000.0


if __name__ == "__main__":
    test_randomize_start_position()

    test_min_distance_to_land()
