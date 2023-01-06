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

import colav_evaluation_tool.common.file_utils as colav_eval_fu
import colav_simulator.common.config_parsing as cp
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.paths as dp  # Default paths
import colav_simulator.core.ship as ship
import matplotlib.pyplot as plt
import numpy as np
import seacharts.enc as senc
import yaml

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
    """Configuration class for a ship scenario.

    If the scenario includes AIS data, one can specify the MMSI of vessels in the ship_list to
    replace predefined AIS trajectories for ships with matching MMSI. This is useful if you
    want to test a COLAV algorithm in an AIS scenario, and put the own-ship in place of a
    predefined ship AIS trajectory in say a Head-on scenario.

    If len(ship_list) < n_ships, the remaining ships will be randomly generated.

    If n_ais_ships from the AIS data is less than n_ships, the remaining ships will be randomly generated.
    """

    name: str
    is_new_scenario: bool
    save_scenario: bool
    type: ScenarioType
    utm_zone: int
    map_data_files: list  # List of file paths to .gdb database files used by seacharts to create the map
    new_load_of_map_data: bool  # If True, seacharts will process .gdb files into shapefiles. If false, it will use existing shapefiles.
    map_size: Optional[Tuple[float, float]] = None  # Size of the map considered in the scenario (in meters) referenced to the origin.
    map_origin_enu: Optional[Tuple[float, float]] = None  # Origin of the map considered in the scenario (in UTM coordinates per now)
    ais_data_file: Optional[Path] = None  # Path to the AIS data file, if considered
    ship_data_file: Optional[Path] = None  # Path to the ship information data file associated with AIS data, if considered
    allowed_nav_statuses: Optional[list] = None  # List of AIS navigation statuses that are allowed in the scenario
    n_random_ships: Optional[int] = 1  # Number of random ships in the scenario, excluding the own-ship, if considered
    min_dist_between_ships: Optional[float] = None  # Used if parts of the scenario are new (randomly generated)
    max_dist_between_ships: Optional[float] = None  # Used if parts of the scenario are new (randomly generated)
    ship_list: Optional[list] = None  # List of ship configurations for the scenario, does not have to be equal to the number of ships in the scenario.

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = ScenarioConfig(
            name=config_dict["name"],
            is_new_scenario=config_dict["is_new_scenario"],
            save_scenario=config_dict["save_scenario"],
            type=ScenarioType(config_dict["type"]),
            utm_zone=config_dict["utm_zone"],
            map_data_files=config_dict["map_data_files"],
            new_load_of_map_data=config_dict["new_load_of_map_data"],
            ship_list=[],
        )

        if "map_size" in config_dict:
            config.map_size = tuple(config_dict["map_size"])

        if "map_origin_enu" in config_dict:
            config.map_origin_enu = tuple(config_dict["map_origin_enu"])

        if "ais_data_file" in config_dict:
            config.ais_data_file = Path(config_dict["ais_data_file"])
            if len(config.ais_data_file.parts) == 1:
                config.ais_data_file = dp.ais_data / config.ais_data_file

            config.ship_data_file = Path(config_dict["ship_data_file"])
            if len(config.ship_data_file.parts) == 1:
                config.ship_data_file = dp.ais_data / config.ship_data_file

            config.allowed_nav_statuses = config_dict["allowed_nav_statuses"]

        if "n_random_ships" in config_dict:
            config.n_random_ships = config_dict["n_random_ships"]

        if "ship_list" in config_dict:
            config.ship_list = []
            for ship_config in config_dict["ship_list"]:
                config.ship_list.append(ship.Config.from_dict(ship_config))

        return config


@dataclass
class Config:
    """Configuration class for managing all parameters/settings related to the creation of scenarios.
    All angle ranges are in degrees, and all distances are in meters.
    """

    n_ship_range: List[int]  # Range of number of random ships to be generated
    n_wps_range: List[int]  # Range of number of waypoints to be generated
    speed_plan_variation_range: List[float]  # Determines maximal +- change in speed plan from one segment to the next
    waypoint_dist_range: List[float]  # Range of [min, max] change in distance between randomly created waypoints
    waypoint_ang_range: List[float]  # Range of [min, max] change in angle between randomly created waypoints
    ho_bearing_range: List[float]  # Range of [min, max] bearing from the own-ship to the target ship for head-on scenarios
    ho_heading_range: List[float]  # Range of [min, max] heading variations of the target ship relative to completely reciprocal head-on scenarios
    ot_bearing_range: List[float]  # Range of [min, max] bearing from the own-ship to the target ship for overtaking scenarios
    ot_heading_range: List[float]  # Range of [min, max] heading variations of the target ship relative to completely parallel overtaking scenarios
    cr_bearing_range: List[float]  # Range of [min, max] bearing from the own-ship to the target ship for crossing scenarios
    cr_heading_range: List[float]  # Range of [min, max] heading variations of the target ship relative to completely orthogonal crossing scenarios


class ScenarioGenerator:
    """Class for generating maritime traffic scenarios in a given geographical environment.

    Internal variables:
        enc (ENC): Electronic Navigational Chart object containing the geographical environment.
        _config (Config): Configuration object containing all parameters/settings related to the creation of scenarios.
    """

    enc: senc.ENC
    _config: Config

    def __init__(self, config_file: Path = dp.scenario_generator_config) -> None:
        """Constructor for the ScenarioGenerator.

        Args:
            config_file (Path, optional): Absolute path to the generator config file. Defaults to dp.scenario_generator_config.
            **kwargs: Keyword arguments for the ScenarioGenerator, can be any of the following:
                    new_data (bool): Flag determining whether or not to read ENC data from shapefiles again.
        """

        self._config = cp.extract(Config, config_file, dp.scenario_generator_schema)

    def _configure_enc(self, scenario_config: ScenarioConfig):
        """Configures the ENC object based on the scenario config file.

        Args:
            scenario_config (ScenarioConfig): Scenario config object.
        """
        print(f"ENC Map size: {scenario_config.map_size}")
        print(f"ENC Map origin: {scenario_config.map_origin_enu}")

        self.enc = senc.ENC(
            config_file=dp.seacharts_config,
            utm_zone=scenario_config.utm_zone,
            size=scenario_config.map_size,
            origin=scenario_config.map_origin_enu,
            files=scenario_config.map_data_files,
            new_data=scenario_config.new_load_of_map_data,
        )
        plt.close()

    def generate(
        self,
        scenario_config_file: Path,
        sample_interval: float = 1.0,
    ) -> Tuple[list, list, ScenarioConfig]:
        """Main class function. Creates a maritime scenario based on the input config file,
        with random plans for each ship unless specified in ship_list entries or loaded from AIS data.

        Args:
            scenario_config_file (Path): Path to the scenario config file.

        Returns:
            Tuple[list, list, ScenarioConfig]: List of ships in the scenario with initialized poses and plans,
                the corresponding ship configuration objects, and the scenario config object.
        """
        print("Generating new scenario...")
        config = cp.extract(ScenarioConfig, scenario_config_file, dp.scenario_schema)

        ais_vessel_data_list = []
        if config.ais_data_file is not None:
            output = colav_eval_fu.read_ais_data(config.ais_data_file, config.ship_data_file, config.utm_zone, config.map_origin_enu, sample_interval)
            ais_vessel_data_list = output["vessels"]
            mmsi_list = output["mmsi_list"]
            config.map_origin_enu = output["origin_enu"]
            config.map_size = output["size"]

        self._configure_enc(config)

        n_cfg_ships = len(config.ship_list)
        ship_list = []
        ship_config_list = []
        pose_list = []
        cfg_ship_idx = 0
        ownship_configured = False
        while ais_vessel_data_list:
            use_ais_ship_trajectory = True
            if cfg_ship_idx < n_cfg_ships:
                ship_config = config.ship_list[cfg_ship_idx]
            else:
                ship_config = ship.Config()

            ship_obj = ship.Ship(mmsi=cfg_ship_idx + 1, config=ship_config)

            # If the mmsi of a ship is specified in the config file,
            # the ship will not use the AIS trajectory, but act
            # on its own (using its onboard planner).
            # Also, the own-ship (with index 0) will not use the predefined AIS trajectory.
            if ship_config.mmsi in mmsi_list:
                use_ais_ship_trajectory = False
                idx = mmsi_list.index(ship_config.mmsi)
            elif cfg_ship_idx == 0:
                ownship_configured = True
                use_ais_ship_trajectory = False
                idx = 0

            ais_vessel = ais_vessel_data_list.pop(idx)
            if ais_vessel.status.value not in config.allowed_nav_statuses:
                continue

            ship_obj.transfer_vessel_ais_data(ais_vessel, use_ais_ship_trajectory)

            if not use_ais_ship_trajectory and ship_config.waypoints is None:
                waypoints = self.generate_random_waypoints(ship_obj.pose[0], ship_obj.pose[1], ship_obj.pose[3], ship_obj.draft)
                speed_plan = self.generate_random_speed_plan(ship_obj.pose[2], U_min=ship_obj.min_speed, U_max=ship_obj.max_speed, n_wps=waypoints.shape[1])
                ship_config.waypoints = waypoints
                ship_config.speed_plan = speed_plan
                ship_obj.set_nominal_plan(waypoints, speed_plan)

            pose_list.append(ship_obj.pose)
            ship_list.append(ship_obj)
            ship_config_list.append(ship_config)

            cfg_ship_idx += 1

        # If the own-ship is not configured, it will be generated randomly
        if not ownship_configured:
            config.n_random_ships += 1

        # The remaining ships are generated randomly
        for i in range(cfg_ship_idx, cfg_ship_idx + config.n_random_ships):
            if cfg_ship_idx < n_cfg_ships:
                ship_config = config.ship_list[i]
            else:
                ship_config = ship.Config()

            ship_obj = ship.Ship(mmsi=i + 1, config=ship_config)

            pose = ship_config.pose
            if ship_config.pose is None:
                if cfg_ship_idx == 0:
                    pose = self.generate_random_pose(ship_obj.max_speed, ship_obj.draft)
                else:
                    pose = self.generate_ts_pose(
                        config.type,
                        pose_list[0],
                        U_min=ship_obj.min_speed,
                        U_max=ship_obj.max_speed,
                        draft=ship_obj.draft,
                    )
                ship_config.pose = pose
                ship_obj.set_initial_state(pose)

            if ship_config.waypoints is None:
                waypoints = self.generate_random_waypoints(pose[0], pose[1], pose[3], ship_obj.draft)
                speed_plan = self.generate_random_speed_plan(pose[2], U_min=ship_obj.min_speed, U_max=ship_obj.max_speed, n_wps=waypoints.shape[1])
                ship_config.waypoints = waypoints
                ship_config.speed_plan = speed_plan
                ship_obj.set_nominal_plan(waypoints, speed_plan)

            pose_list.append(pose)
            ship_list.append(ship_obj)
            ship_config_list.append(ship_config)

        # if config.save_scenario:
        # append string of date and time for scenario creation
        # save_scenario(ship_config_list, scenario_config_file / ".yaml")

        return ship_list, ship_config_list, config

    def generate_ts_pose(
        self,
        scenario_type: ScenarioType,
        os_pose: np.ndarray,
        U_min: float = 1.0,
        U_max: float = 15.0,
        draft: float = 2.0,
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
            draft (float, optional): Draft of target ship. Defaults to 2.0.
            min_dist_between_ships (float, optional): Minimum distance between own-ship and target ship. Defaults to 100.0.
            max_dist_between_ships (float, optional): Maximum distance between own-ship and target ship. Defaults to 1000.0.
            min_land_clearance (float, optional): Minimum distance between target ship and land. Defaults to 100.0.

        Returns:
            Tuple[float, float]: Target ship position = [x, y].
        """

        if any(np.isnan(os_pose)):
            return self.generate_random_pose(max_speed=U_max, draft=draft)

        if scenario_type == ScenarioType.HO:
            bearing = random.uniform(self._config.ho_bearing_range[0], self._config.ho_bearing_range[1])
            speed = random.uniform(U_min, U_max)
            heading_modifier = 180.0 + random.uniform(self._config.ho_heading_range[0], self._config.ho_heading_range[1])

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
            heading_modifier = -90.0 + random.uniform(self._config.cr_heading_range[0], self._config.cr_heading_range[1])

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

    def generate_random_waypoints(self, x: float, y: float, psi: float, draft: float = 5.0, n_wps: Optional[int] = None) -> np.ndarray:
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

                distance_wp_to_wp = random.uniform(self._config.waypoint_dist_range[0], self._config.waypoint_dist_range[1])
                distance_wp_to_wp = mf.sat(distance_wp_to_wp, 0.0, min_dist_to_land)

                alpha = 0.0
                if i > 1:
                    alpha = np.deg2rad(random.uniform(self._config.waypoint_ang_range[0], self._config.waypoint_ang_range[1]))

                new_wp = np.array(
                    [
                        waypoints[0, i - 1] + distance_wp_to_wp * np.cos(psi + alpha),
                        waypoints[1, i - 1] + distance_wp_to_wp * np.sin(psi + alpha),
                    ],
                )

                crosses_grounding_hazards = mapf.check_if_segment_crosses_grounding_hazards(self.enc, new_wp, waypoints[:, i - 1], draft)

                if cgh_count >= 10:
                    break

            if cgh_count >= 10:
                waypoints = waypoints[:, 0:i]
                break

            waypoints[:, i] = new_wp

        return waypoints

    def generate_random_speed_plan(self, U: float, U_min: float = 1.0, U_max: float = 15.0, n_wps: Optional[int] = None) -> np.ndarray:
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
            U_mod = random.uniform(self._config.speed_plan_variation_range[0], self._config.speed_plan_variation_range[1])
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
