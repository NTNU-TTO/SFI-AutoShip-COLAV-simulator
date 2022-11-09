"""
    scenario_generator.py

    Summary:
        Contains a class for generating scenarios for the simulator.

    Author: Trym Tengesdal, Joachim Miller, Melih Akdag
"""

import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import colav_simulator.common.config_parsing as config_parsing
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.paths as dp  # Default paths
import colav_simulator.ships.ship as ship
import numpy as np
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
class ScenarioConfig:
    """Configuration class for specifying a new scenario."""

    type: ScenarioType
    n_ships: int
    min_dist_between_ships: float
    ship_list: List[ship.Config]


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
        _enc (ENC): Electronic Navigational Chart object containing the geographical environment.
        _config (Config): Configuration object containing all parameters/settings related to the creation of scenarios.
    """

    _enc: ENC
    _config: Config

    def __init__(
        self, enc_config_file: Path = dp.seacharts_config, config_file: Path = dp.scenario_generator_config
    ) -> None:
        self._enc = ENC(enc_config_file, new_data=False)
        self._config = config_parsing.extract(Config, config_file, dp.scenario_generator_schema)

    def generate(
        self,
        scenario_config_file: Path = dp.new_scenario_config,
    ) -> Tuple[list, list, list]:
        """Creates a maritime scenario based on the input config file, with random plans for each ship
        unless specified.

        Args:
            scenario_config_file (Path): Path to the scenario config file.

        Returns:
            List[Ship]: List of ships in the scenario with initialized poses and plans.
        """

        scenario_config = config_parsing.extract(ScenarioConfig, scenario_config_file, dp.new_scenario_schema)
        scenario_config = convert_scenario_config_dict_to_dataclass(scenario_config)

        ship_list = []

        n_ships = scenario_config.n_ships
        n_configured_ships = len(ship_list)

        for i in range(n_ships):

            if i == 0:
                pose = self._generate_random_pose(ship_list[0].model.pars.U_max)
            pose_list = [os_pose.tolist()]

            pose = self._generate_ts_pose(scenario_type, os_pose, U_max=ts_max_speed)
            pose_list.append(ts_pose.tolist())

        # Create plan (waypoints, speed_plan) for all ships which does not have it
        waypoint_list = []
        speed_plan_list = []
        for i in range(num_ships):
            waypoints = self._generate_random_waypoints(pose_list[i][0], pose_list[i][1], pose_list[i][3])
            waypoint_list.append(waypoints)

            speed_plan = self._generate_random_speed_plan(pose_list[i][2])
            speed_plan_list.append(speed_plan)

        return ship_list

    def _generate_ts_pose(
        self, scenario_type: ScenarioType, os_pose: np.ndarray, U_min: float = 1.0, U_max: float = 15.0
    ) -> np.ndarray:
        """Generates a position for the target ship based on the perspective/pose of the first ship/own-ship,
        such that the scenario is of the input type.

        Args:
            scenario_type (ScenarioType): Type of scenario.
            os_pose (np.ndarray): Own-ship pose = [x, y, speed, heading].
            U_min (float, optional): Minimum speed. Defaults to 1.0.
            U_max (float, optional): Maximum speed. Defaults to 15.0.

        Returns:
            Tuple[float, float]: Target ship position = [x, y].
        """

        distance_to_land = mapf.min_distance_to_land(self._enc, os_pose[1], os_pose[0])
        distance_os_ts = random.uniform(50.0, distance_to_land)

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

        x = os_pose[0] + distance_os_ts * np.cos(os_pose[3] + bearing)
        y = os_pose[1] + distance_os_ts * np.sin(os_pose[3] + bearing)
        return np.array([x, y, speed, heading])

    def _generate_random_pose(
        self, max_speed: float = 15.0, draft: float = 5.0, heading: Optional[float] = None
    ) -> np.ndarray:
        """Creates a random pose which adheres to the ship's draft and maximum speed.

        Args:
            max_speed (float): Vessel's maximum speed
            draft (float, optional): How deep the ship keel is into the water. Defaults to 5.
            heading (Optional[float], optional): Heading of the ship in radians. Defaults to None.

        Returns:
            np.ndarray: Array containing the vessel pose = [x, y, speed, heading]
        """
        x, y = mapf.randomize_start_position_from_draft(self._enc, draft)

        speed = random.uniform(0.0, max_speed)

        if heading is None:
            heading = random.uniform(0, 2.0 * np.pi)

        return np.array([x, y, speed, heading])

    def _generate_random_waypoints(self, x: float, y: float, psi: float, n_wps: Optional[int] = None) -> np.ndarray:
        """Creates random waypoints starting from a ship position and heading.

        Args:
            x (float): x position (north) of the ship.
            y (float): y position (east) of the ship.
            psi (float): heading of the ship in radians.
            n_wps (Optional[int]): Number of waypoints to create.

        Returns:
            np.ndarray: 2 x n_wps array of waypoints.
        """
        if n_wps is None:
            n_wps = random.randint(self._config.n_wps_range[0], self._config.n_wps_range[1])

        waypoints = np.array((2, n_wps))
        waypoints[:, 0] = np.array([x, y])
        for i in range(1, n_wps):
            crosses_grounding_hazards = True
            while crosses_grounding_hazards:
                distance_wp_to_wp = random.uniform(
                    self._config.waypoint_dist_range[0], self._config.waypoint_dist_range[1]
                )
                alpha = np.deg2rad(
                    random.uniform(self._config.waypoint_ang_range[0], self._config.waypoint_ang_range[1])
                )

                new_wp = np.array(
                    [waypoints[0, i - 1] + distance_wp_to_wp * np.cos(psi + alpha)],
                    [waypoints[1, i - 1] + distance_wp_to_wp * np.sin(psi + alpha)],
                )

                crosses_grounding_hazards = mapf.check_if_segment_crosses_grounding_hazards(
                    self._enc, new_wp, waypoints[:, i - 1]
                )

            waypoints[:, i] = new_wp

        return waypoints

    def _generate_random_speed_plan(
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

        speed_plan = np.array(n_wps)
        speed_plan[0] = U
        for i in range(1, n_wps):
            lb = max(U_min, speed_plan[i - 1] - self._config.speed_plan_variation_range[0])
            ub = min(U_max, speed_plan[i - 1] + self._config.speed_plan_variation_range[1])
            U = U + random.uniform(lb, ub)
            speed_plan[i] = U

        return speed_plan


def convert_scenario_config_dict_to_dataclass(config_dict: dict) -> ScenarioConfig:
    """Converts a dictionary to a ScenarioConfig dataclass.

    Args:
        scenario_config_dict (dict): Dictionary containing the scenario config.

    Returns:
        ScenarioConfig: ScenarioConfig dataclass.
    """
    config = ScenarioConfig(
        type=ScenarioType(config_dict["type"]),
        n_ships=config_dict["n_ships"],
        min_dist_between_ships=config_dict["min_dist_between_ships"],
        ship_list=[],
    )

    for ship_config_dict in config_dict["ship_list"]:
        ship_config = ship.convert_ship_config_dict_to_dataclass(ship_config_dict)

        config.ship_list.append(ship_config)
    return config
