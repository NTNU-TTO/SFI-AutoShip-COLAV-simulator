import math
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import colav_simulator.common.file_utils as futils
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.paths as dp  # Default paths
import numpy as np
from cerberus import Validator
from seacharts.enc import ENC

np.set_printoptions(suppress=True, formatter={"float_kind": "{:.2f}".format})


class ScenarioType(Enum):
    """Enum for the different possible scenario/situation types."""

    RANDOM = 0
    HO = 1
    OT_ing = 2
    OT_en = 3
    CR_GW = 4
    CR_SO = 5
    MS = 6


@dataclass
class Config:
    """Configuration class for managing all parameters/settings related to the creation of scenarios."""

    _schema: dict
    enc: ENC

    def __init__(self, config_file: Path = dp.scenario_generation_config, **kwargs) -> None:
        self._schema = futils.read_yaml_into_dict(dp.scenario_generation_schema)
        self.validator = Validator(self._schema)

        self.parse(config_file)
        self.override(**kwargs)

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, new_settings: dict):
        self._settings = new_settings

    def validate(self, settings: dict) -> None:
        if not settings:
            raise ValueError("Empty settings!")

        if not self._schema:
            raise ValueError("Empty schema!")

        if not self.validator.validate(settings):
            raise ValueError(f"Cerberus validation Error: {self.validator.errors}")

    def parse(self, file_name: Path) -> None:
        self._settings = futils.read_yaml_into_dict(file_name)
        self.validate(self._settings)

    def override(self, **kwargs) -> None:
        if not kwargs:
            return

        new_settings = self._settings
        for key, value in kwargs.items():
            new_settings[key] = value

        self.validate(new_settings)

        self._settings = new_settings


class ScenarioGenerator:
    """Class for generating maritime traffic scenarios in a given geographical environment.

    Internal variables:
        enc (ENC): Electronic Navigational Chart object containing the geographical environment.

    """

    _config: Config

    def __init__(self, enc: ENC, config_file: Optional[Path] = None) -> None:
        if config_file:
            self._config = Config(config_file)
        else:
            self._config = Config()
        self._enc = enc

    def generate(
        self, os_max_speed: float = 15.0, ts_max_speed: float = 15.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Creates a COLREGS scenario based on the scenario type, with random plans for all ships

        Args:
            num_ships (int): Number of ships in the scenario

        Returns:
            dict: Dictionary containing the scenario
        """
        # Create the initial poses
        x1, y1, speed1, heading1, x2, y2, speed2, heading2 = random_scenario_generator(
            scenario_num, os_max_speed, ts_max_speed, ship_model_name_list[0]
        )
        pose_list = [[x1, y1, speed1, heading1], [x2, y2, speed2, heading2]]
        for i in range(2, num_ships):
            x, y, speed, heading = random_pose(ts_max_speed, ship_model_name=ship_model_name_list[i])
            pose_list.append([x, y, speed, heading])

        # Create plan (waypoints, speed_plan) for all ships
        waypoint_list = []
        speed_plan_list = []
        for i in range(num_ships):
            wp = create_random_waypoints(pose_list[i][0], pose_list[i][1], pose_list[i][3], n_wps)
            speed_plan = create_random_speed_plan(pose_list[i][2], n_wps)
            waypoint_list.append(wp)
            speed_plan_list.append(speed_plan)
        return pose_list, waypoint_list, speed_plan_list

    def generate_two_ship_scenario(scenario_num, os_max_speed, ts_max_speed, ship_model_name):
        """Generate random COLREGS scenario based on type and ship max speeds.

        Args:

        """
        if scenario_num == 0:
            n = random.randint(1, 5)
            x1, y1, speed1, heading1, x2, y2, speed2, heading2 = random_scenario_generator(
                n, os_max_speed, ts_max_speed, ship_model_name
            )
        elif scenario_num == 1:
            x1, y1, speed1, heading1, x2, y2, speed2, heading2 = head_on(os_max_speed, ts_max_speed, ship_model_name)
        elif scenario_num == 2:
            x1, y1, speed1, heading1, x2, y2, speed2, heading2 = overtaking(os_max_speed, ship_model_name)
        elif scenario_num == 3:
            x1, y1, speed1, heading1, x2, y2, speed2, heading2 = overtaken(os_max_speed, ship_model_name)
        elif scenario_num == 4:
            x1, y1, speed1, heading1, x2, y2, speed2, heading2 = crossing_give_way(
                os_max_speed, ts_max_speed, ship_model_name
            )
        elif scenario_num == 5:
            x1, y1, speed1, heading1, x2, y2, speed2, heading2 = crossing_stand_on(
                os_max_speed, ts_max_speed, ship_model_name
            )

        return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def create_scenario(num_ships, scenario_num, ship_model_name_list, os_max_speed, ts_max_speed, n_wps: int):
    """Creates a COLREGS scenario based on the scenario type, with random plans for all ships

    Args:
        num_ships (int): Number of ships in the scenario

    Returns:
        dict: Dictionary containing the scenario
    """
    # Create the initial poses
    x1, y1, speed1, heading1, x2, y2, speed2, heading2 = random_scenario_generator(
        scenario_num, os_max_speed, ts_max_speed, ship_model_name_list[0]
    )
    pose_list = [[x1, y1, speed1, heading1], [x2, y2, speed2, heading2]]
    for i in range(2, num_ships):
        x, y, speed, heading = random_pose(ts_max_speed, ship_model_name=ship_model_name_list[i])
        pose_list.append([x, y, speed, heading])

    # Create plan (waypoints, speed_plan) for all ships
    waypoint_list = []
    speed_plan_list = []
    for i in range(num_ships):
        wp = create_random_waypoints(pose_list[i][0], pose_list[i][1], pose_list[i][3], n_wps)
        speed_plan = create_random_speed_plan(pose_list[i][2], n_wps)
        waypoint_list.append(wp)
        speed_plan_list.append(speed_plan)
    return pose_list, waypoint_list, speed_plan_list


def random_pose(enc: ENC, max_speed: float = 15.0, draft: float = 5.0) -> np.ndarray:
    """Creates a random pose (x, y, speed, heading)

    Args:
        max_speed (float): Vessel's maximum speed
        draft (float, optional): How deep the ship keel is into the water. Defaults to 5.

    Returns:
        Tuple[float, float, float, float]: Tuple containing the vessel state
    """
    x, y = mapf.randomize_start_position_from_draft(enc, draft)

    speed = round(random.uniform(1, max_speed), 1)

    heading = random.uniform(0, 2.0 * np.pi)

    return np.array([x, y, speed, heading])


def create_random_waypoints(enc: ENC, x: float, y: float, psi: float, n_wps: int) -> np.ndarray:
    """Creates random waypoints starting from a ship position and heading.

    Args:
        x (float): x position (north) of the ship.
        y (float): y position (east) of the ship.
        psi (float): heading of the ship in radians.
        n_wps (int): Number of waypoints to create.

    Returns:
        np.ndarray: 2 x n_wps array of waypoints.
    """
    waypoints = np.array((2, n_wps))
    waypoints[:, 0] = np.array([x, y])
    for i in range(1, n_wps):
        crosses_grounding_hazards = True
        while crosses_grounding_hazards:
            distance_wp_to_wp = random.randint(200, 1000)
            alpha = random.uniform(0, np.pi / 4)

            new_wp = np.array(
                [waypoints[0, i - 1] + distance_wp_to_wp * np.cos(psi + alpha)],
                [waypoints[1, i - 1] + distance_wp_to_wp * np.sin(psi + alpha)],
            )

            crosses_grounding_hazards = mapf.check_if_segment_crosses_grounding_hazards(
                enc, new_wp, waypoints[:, i - 1]
            )

        waypoints[:, i] = new_wp
    return waypoints


def create_random_speed_plan(U_min: float, U_max: float, n_wps: int) -> np.ndarray:
    """Creates a random speed plan using the input minimum and maximum speed.

    Args:
        U_min (float): Minimum speed.
        U_max (float): Maximum speed.
        n_wps (int): Number of waypoints to create.

    Returns:
        np.ndarray: 1 x n_wps array containing the speed plan.
    """
    speed_plan = np.array(n_wps)
    U = random.uniform(U_min, U_max)
    speed_plan[0] = U
    for i in range(1, n_wps):
        U = U + random.uniform(-1.0, 1.0)
        speed_plan[i] = U
    return speed_plan


def head_on(enc: ENC, os_max_speed, ts_max_speed, ship_model_name):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(os_max_speed, ship_model_name)

    # random target ship considering own ship pose
    distance_land = mapf.min_distance_to_land(enc, y1, x1)
    x2 = x1 + distance_land * math.cos(math.radians(heading1))
    y2 = y1 + distance_land * math.sin(math.radians(heading1))
    speed2 = round(random.uniform(1, ts_max_speed), 1)
    heading2 = heading1 + 180 + random.uniform(-14, 14)

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def overtaking(enc: ENC, os_max_speed, ship_model_name):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(os_max_speed, ship_model_name)

    # random target ship considering own ship pose
    distance_land = mapf.min_distance_to_land(enc, y1, x1)
    x2 = x1 + distance_land * math.cos(math.radians(heading1))
    y2 = y1 + distance_land * math.sin(math.radians(heading1))
    speed2 = round((speed1 - speed1 * random.uniform(0.5, 0.9)), 1)
    heading2 = heading1 + random.uniform(-13, 13)

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def overtaken(enc: ENC, os_max_speed, ship_model_name):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(os_max_speed, ship_model_name)

    # random target ship considering own ship pose
    distance_land = mapf.min_distance_to_land(enc, y1, x1)
    x2 = x1 - distance_land * math.cos(math.radians(heading1))
    y2 = y1 - distance_land * math.sin(math.radians(heading1))
    speed2 = round((speed1 + speed1 * random.uniform(0.5, 0.9)), 1)
    heading2 = heading1 + random.uniform(-13, 13)

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def crossing_give_way(enc: ENC, os_max_speed, ts_max_speed, ship_model_name):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(os_max_speed, ship_model_name)

    # random target ship considering own ship pose
    n = random.uniform(0, 112.5)
    distance_land = mapf.min_distance_to_land(enc, y1, x1)
    x2 = x1 + distance_land * math.cos(math.radians(heading1 + n))
    y2 = y1 + distance_land * math.sin(math.radians(heading1 + n))
    speed2 = round(random.uniform(1, ts_max_speed), 1)
    heading2 = heading1 - 90

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def crossing_stand_on(enc: ENC, os_max_speed, ts_max_speed, ship_model_name):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(os_max_speed, ship_model_name)

    # random target ship considering own ship pose
    n = random.uniform(-112.5, 0)
    distance_to_land = mapf.min_distance_to_land(enc, y1, x1)
    x2 = x1 + distance_to_land * np.cos(np.radians(heading1 + n))
    y2 = y1 + distance_to_land * np.sin(np.radians(heading1 + n))
    speed2 = round(random.randint(1, ts_max_speed), 1)
    heading2 = heading1 + 90

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2
