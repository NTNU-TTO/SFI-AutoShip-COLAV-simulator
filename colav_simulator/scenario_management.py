"""
    scenario_management.py

    Summary:
        Contains functionality for loading existing scenario definitions,
        and also a ScenarioGenerator class for generating new scenarios. Functionality
        for saving these new scenarios also exists.

    Author: Trym Tengesdal, Joachim Miller, Melih Akdag
"""

import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import colav_simulator.common.config_parsing as config_parsing
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.paths as dp  # Default paths
import colav_simulator.core.ship as ship
import numpy as np
import yaml
from seacharts.enc import ENC

np.set_printoptions(suppress=True, formatter={"float_kind": "{:.2f}".format})


class ScenarioType(Enum):
    """Enum for the different possible scenario/situation types.

    Explanation:
        SS: Only one ship (the own-ship) in the scenario.
        HO: Head on scenario.
        OT_ing: Overtaking scenario (own-ship overtakes and should give-way).
        OT_en: Overtaken scenario (own-ship is overtaken and should stand-on).
        CR_GW: Crossing scenario where own-ship has give-way duties.
        CR_SO: Crossing scenario where own-ship has stand-on duties.
        RANDOM: Random number of ships in the scenario.
    """

    SS = 0
    HO = 1
    OT_ing = 2
    OT_en = 3
    CR_GW = 4
    CR_SO = 5
    RANDOM = 6


@dataclass
class ExistingScenarioConfig:
    """Configuration class for existing scenarios."""

    ship_list: list

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = ExistingScenarioConfig(
            ship_list=[],
        )

        for ship_config in config_dict["ship_list"]:
            config.ship_list.append(ship.Config.from_dict(ship_config))

        return config


@dataclass
class NewScenarioConfig:
    """Configuration class for specifying a new scenario."""

    type: ScenarioType
    n_ships: int
    min_dist_between_ships: float
    ship_list: List[ship.Config]

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = NewScenarioConfig(
            type=ScenarioType(config_dict["type"]),
            n_ships=config_dict["n_ships"],
            min_dist_between_ships=config_dict["min_dist_between_ships"],
            ship_list=[],
        )

        for ship_config in config_dict["ship_list"]:
            config.ship_list.append(ship.Config.from_dict(ship_config))

        return config


@dataclass
class Config:
    """Configuration class for managing all parameters/settings related to the creation of scenarios.
    All angle ranges are in degrees, and all distances are in meters.
    """

    n_ship_range: List[int]  # Range of number of ships to be generated
    n_wps_range: List[int]  # Range of number of waypoints to be generated
    speed_plan_variation_range: List[float]  # Determines maximal +- change in speed plan from one segment to the next
    waypoint_dist_range: List[float]  # Range of [min, max] change in distance between randomly created waypoints
    waypoint_ang_range: List[float]  # Range of [min, max] change in angle between randomly created waypoints
    ho_bearing_range: List[
        float
    ]  # Range of [min, max] bearing from the own-ship to the target ship for head-on scenarios
    ho_heading_range: List[
        float
    ]  # Range of [min, max] heading variations of the target ship relative to completely reciprocal head-on scenarios
    ot_bearing_range: List[
        float
    ]  # Range of [min, max] bearing from the own-ship to the target ship for overtaking scenarios
    ot_heading_range: List[
        float
    ]  # Range of [min, max] heading variations of the target ship relative to completely parallel overtaking scenarios
    cr_bearing_range: List[
        float
    ]  # Range of [min, max] bearing from the own-ship to the target ship for crossing scenarios
    cr_heading_range: List[
        float
    ]  # Range of [min, max] heading variations of the target ship relative to completely orthogonal crossing scenarios


class ScenarioGenerator:
    """Class for generating maritime traffic scenarios in a given geographical environment.

    Internal variables:
        enc (ENC): Electronic Navigational Chart object containing the geographical environment.
        _config (Config): Configuration object containing all parameters/settings related to the creation of scenarios.
    """

    enc: ENC
    _config: Config

    def __init__(
        self, enc_config_file: Path = dp.seacharts_config, config_file: Path = dp.scenario_generator_config
    ) -> None:
        """Constructor for the ScenarioGenerator.

        Args:
            enc_config_file (Path, optional): Absolute path to the ENC config file. Defaults to dp.seacharts_config.
            config_file (Path, optional): Absolute path to the generator config file. Defaults to dp.scenario_generator_config.
            **kwargs: Keyword arguments for the ScenarioGenerator, can be any of the following:
                    new_data (bool): Flag determining whether or not to read ENC data from shapefiles again.
        """
        self.enc = ENC(enc_config_file)

        self._config = config_parsing.extract(Config, config_file, dp.scenario_generator_schema)

    def generate(
        self,
        scenario_config_file: Path = dp.new_scenario_config,
    ) -> Tuple[list, list]:
        """Creates a maritime scenario based on the input config file, with random plans for each ship
        unless specified.

        Args:
            scenario_config_file (Path): Path to the scenario config file.

        Returns:
            Tuple[list, list]: List of ships in the scenario with initialized poses and plans,
                and also the corresponding ship configuration objects.
        """
        print("Generating new scenario...")
        config = config_parsing.extract(NewScenarioConfig, scenario_config_file, dp.new_scenario_schema)

        n_ships = config.n_ships
        n_cfg_ships = len(config.ship_list)

        ship_list = []
        ship_config_list = []
        pose_list = []
        for i in range(n_ships):
            if i < n_cfg_ships:
                ship_config = config.ship_list[i]
            else:
                ship_config = ship.Config()

            ship_obj = ship.Ship(mmsi=i + 1, config=ship_config)

            if i == 0:  # Own-ship
                pose = self.generate_random_pose(ship_obj.max_speed, ship_obj.draft)
            else:  # Target ships
                pose = self.generate_ts_pose(
                    config.type, pose_list[0], U_min=ship_obj.min_speed, U_max=ship_obj.max_speed
                )

            waypoints = self.generate_random_waypoints(pose[0], pose[1], pose[3], ship_obj.draft)
            speed_plan = self.generate_random_speed_plan(
                pose[2], U_min=ship_obj.min_speed, U_max=ship_obj.max_speed, n_wps=waypoints.shape[1]
            )

            ship_obj.set_initial_state(pose)
            ship_obj.set_nominal_plan(waypoints, speed_plan)

            pose_list.append(pose)
            ship_list.append(ship_obj)

            ship_config.pose = pose
            ship_config.waypoints = waypoints
            ship_config.speed_plan = speed_plan
            ship_config_list.append(ship_config)

        return ship_list, ship_config_list

    def generate_ts_pose(
        self,
        scenario_type: ScenarioType,
        os_pose: np.ndarray,
        U_min: float = 1.0,
        U_max: float = 15.0,
        min_dist_between_ships: float = 100.0,
        max_dist_between_ships: float = 1000.0,
        min_land_clearance: float = 100.0,
    ) -> np.ndarray:
        """Generates a position for the target ship based on the perspective/pose of the first ship/own-ship,
        such that the scenario is of the input type.

        Args:
            scenario_type (ScenarioType): Type of scenario.
            os_pose (np.ndarray): Own-ship pose = [x, y, speed, heading].
            U_min (float, optional): Minimum speed. Defaults to 1.0.
            U_max (float, optional): Maximum speed. Defaults to 15.0.
            min_dist_between_ships (float, optional): Minimum distance between own-ship and target ship. Defaults to 100.0.
            min_land_clearance (float, optional): Minimum distance between target ship and land. Defaults to 100.0.

        Returns:
            Tuple[float, float]: Target ship position = [x, y].
        """

        if scenario_type == ScenarioType.HO:
            bearing = random.uniform(self._config.ho_bearing_range[0], self._config.ho_bearing_range[1])
            speed = random.uniform(U_min, U_max)
            heading_modifier = 180.0 + random.uniform(
                self._config.ho_heading_range[0], self._config.ho_heading_range[1]
            )

        elif scenario_type == ScenarioType.OT_ing:
            assert U_min < os_pose[2]  # Own-ship speed must be greater than the minimum target ship speed.
            bearing = random.uniform(self._config.ot_bearing_range[0], self._config.ot_bearing_range[1])
            speed = random.uniform(U_min, os_pose[2])
            heading_modifier = random.uniform(self._config.ot_heading_range[0], self._config.ot_heading_range[1])

        elif scenario_type == ScenarioType.OT_en:
            assert U_max > os_pose[2]  # Own-ship speed must be less than the maximum target ship speed.
            bearing = random.uniform(self._config.ot_bearing_range[0], self._config.ot_bearing_range[1])
            speed = random.uniform(os_pose[2], U_max)
            heading_modifier = random.uniform(self._config.ot_heading_range[0], self._config.ot_heading_range[1])

        elif scenario_type == ScenarioType.CR_GW:
            bearing = random.uniform(self._config.cr_bearing_range[0], self._config.cr_bearing_range[1])
            speed = random.uniform(U_min, U_max)
            heading_modifier = -90.0 + random.uniform(
                self._config.cr_heading_range[0], self._config.cr_heading_range[1]
            )

        elif scenario_type == ScenarioType.CR_SO:
            bearing = random.uniform(self._config.cr_bearing_range[0], self._config.cr_bearing_range[1])
            speed = random.uniform(U_min, U_max)
            heading_modifier = 90.0 + random.uniform(self._config.cr_heading_range[0], self._config.cr_heading_range[1])

        else:
            bearing = random.uniform(0.0, 2.0 * np.pi)
            speed = random.uniform(U_min, U_max)
            heading_modifier = random.uniform(0.0, 359.999)

        bearing = np.deg2rad(bearing)
        heading = os_pose[3] + np.deg2rad(heading_modifier)

        is_safe_pose = False
        while not is_safe_pose:
            distance_os_ts = random.uniform(min_dist_between_ships, max_dist_between_ships)
            x = os_pose[0] + distance_os_ts * np.cos(os_pose[3] + bearing)
            y = os_pose[1] + distance_os_ts * np.sin(os_pose[3] + bearing)

            distance_to_land = mapf.min_distance_to_land(self.enc, y, x)

            if distance_to_land >= min_land_clearance:
                is_safe_pose = True

        return np.array([x, y, speed, heading])

    def generate_random_pose(
        self,
        max_speed: float = 15.0,
        draft: float = 5.0,
        heading: Optional[float] = None,
        land_clearance: float = 100.0,
    ) -> np.ndarray:
        """Creates a random pose which adheres to the ship's draft and maximum speed.

        Args:
            max_speed (float): Vessel's maximum speed
            draft (float, optional): How deep the ship keel is into the water. Defaults to 5.
            heading (Optional[float], optional): Heading of the ship in radians. Defaults to None.

        Returns:
            np.ndarray: Array containing the vessel pose = [x, y, speed, heading]
        """
        x, y = mapf.generate_random_start_position_from_draft(self.enc, draft, land_clearance)

        speed = random.uniform(0.0, max_speed)

        if heading is None:
            heading = random.uniform(0.0, 2.0 * np.pi)

        return np.array([x, y, speed, heading])

    def generate_random_waypoints(
        self, x: float, y: float, psi: float, draft: float = 5.0, n_wps: Optional[int] = None
    ) -> np.ndarray:
        """Creates random waypoints starting from a ship position and heading.

        Args:
            x (float): x position (north) of the ship.
            y (float): y position (east) of the ship.
            psi (float): heading of the ship in radians.
            n_wps (Optional[int]): Number of waypoints to create.
            no_enc (bool): If True, the waypoints will not be constrained by the ENC area.

        Returns:
            np.ndarray: 2 x n_wps array of waypoints.
        """
        if n_wps is None:
            n_wps = random.randint(self._config.n_wps_range[0], self._config.n_wps_range[1])

        waypoints = np.zeros((2, n_wps))
        waypoints[:, 0] = np.array([x, y])
        for i in range(1, n_wps):
            min_dist_to_land = mapf.min_distance_to_land(self.enc, waypoints[1, i - 1], waypoints[0, i - 1])
            crosses_grounding_hazards = True
            cgh_count = -1
            while crosses_grounding_hazards:
                cgh_count += 1

                distance_wp_to_wp = random.uniform(
                    self._config.waypoint_dist_range[0], self._config.waypoint_dist_range[1]
                )
                distance_wp_to_wp = mf.sat(distance_wp_to_wp, 0.0, min_dist_to_land)

                alpha = 0.0
                if i > 1:
                    alpha = np.deg2rad(
                        random.uniform(self._config.waypoint_ang_range[0], self._config.waypoint_ang_range[1])
                    )

                new_wp = np.array(
                    [
                        waypoints[0, i - 1] + distance_wp_to_wp * np.cos(psi + alpha),
                        waypoints[1, i - 1] + distance_wp_to_wp * np.sin(psi + alpha),
                    ],
                )

                crosses_grounding_hazards = mapf.check_if_segment_crosses_grounding_hazards(
                    self.enc, new_wp, waypoints[:, i - 1], draft
                )

                if cgh_count >= 10:
                    break

            if cgh_count >= 10:
                waypoints = waypoints[:, 0:i]
                break

            waypoints[:, i] = new_wp

        return waypoints

    def generate_random_speed_plan(
        self, U: float, U_min: float = 1.0, U_max: float = 15.0, n_wps: Optional[int] = None
    ) -> np.ndarray:
        """Creates a random speed plan using the input speed and min/max speed of the ship.

        Args:
            U (float): The ship's speed.
            U_min (float, optional): The ship's minimum speed. Defaults to 1.0.
            U_max (float, optional): The ship's maximum speed. Defaults to 15.0.
            n_wps (Optional[int]): Number of waypoints to create.

        Returns:
            np.ndarray: 1 x n_wps array containing the speed plan.
        """
        if n_wps is None:
            n_wps = random.randint(self._config.n_wps_range[0], self._config.n_wps_range[1])

        speed_plan = np.zeros(n_wps)
        speed_plan[0] = U
        for i in range(1, n_wps):
            U_mod = random.uniform(
                self._config.speed_plan_variation_range[0], self._config.speed_plan_variation_range[1]
            )
            speed_plan[i] = mf.sat(speed_plan[i - 1] + U_mod, U_min, U_max)

            if i == n_wps - 1:
                speed_plan[i] = 0.0

        return speed_plan

    @property
    def enc_bbox(self) -> np.ndarray:
        """Returns the bounding box of the considered ENC area.

        Returns:
            np.ndarray: Array containing the ENC bounding box = [min_x, min_y, max_x, max_y]
        """
        size = self.enc.size
        origin = self.enc.origin

        return np.array([origin[0], origin[1], origin[0] + size[0], origin[1] + size[1]])

    @property
    def enc_origin(self) -> np.ndarray:
        return np.array([self.enc.origin[1], self.enc.origin[0]])


def load_scenario_definition(scenario_file: Path):
    """
    Loads a scenario definition from a yaml file and processes into a list ships with specified poses,
    waypoints and speed plans (the definition). The ships are configured as specified in the scenario file.

    Parameters:
        scenario_file (Path): Absolute path to scenario file.

    Returns:
        list: List of configured ships with specified poses, waypoints and speed plans.
    """
    print("Loading existing scenario definition...")
    ship_list = []

    config = config_parsing.extract(ExistingScenarioConfig, scenario_file, dp.existing_scenario_schema)

    for i, ship_config in enumerate(config.ship_list):
        ship_list.append(ship.Ship(mmsi=i + 1, config=ship_config))

    return ship_list


def save_scenario(ship_config_list: list, save_file: Path) -> None:
    """Saves the the scenario defined by the list of configured ships, to a json file as a dict at savefile

    Args:
        ship_config_list (list): List of ship configurations.
        save_file (Path): Absolute path to save the scenario definition to.
    """

    scenario_data_dict: dict = {}
    scenario_data_dict["ship_list"] = []
    for ship_config in ship_config_list:
        ship_data_dict = ship_config.to_dict()
        scenario_data_dict["ship_list"].append(ship_data_dict)

    with save_file.open(mode="w") as file:
        yaml.dump(scenario_data_dict, file)
