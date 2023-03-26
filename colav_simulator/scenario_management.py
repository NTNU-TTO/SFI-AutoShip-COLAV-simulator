"""
    scenario_management.py

    Summary:
        Contains functionality for loading existing scenario definitions,
        and also a ScenarioGenerator class for generating new scenarios. Functionality
        for saving these new scenarios also exists.

    Author: Trym Tengesdal, Joachim Miller, Melih Akdag
"""

import copy
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import colav_evaluation_tool.common.file_utils as colav_eval_fu
import colav_simulator.common.config_parsing as cp
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp  # Default paths
import colav_simulator.core.ship as ship
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
        MS: Multiple ships scenario without any specification of COLREGS situations.
    """

    SS = 0
    HO = 1
    OT_ing = 2
    OT_en = 3
    CR_GW = 4
    CR_SO = 5
    MS = 6


@dataclass
class ScenarioConfig:
    """Configuration class for a ship scenario."""

    name: str
    save_scenario: bool
    t_start: float
    t_end: float
    dt_sim: float
    type: ScenarioType
    utm_zone: int
    map_data_files: list  # List of file paths to .gdb database files used by seacharts to create the map
    new_load_of_map_data: bool  # If True, seacharts will process .gdb files into shapefiles. If false, it will use existing shapefiles.
    map_size: Optional[Tuple[float, float]] = None  # Size of the map considered in the scenario (in meters) referenced to the origin.
    map_origin_enu: Optional[Tuple[float, float]] = None  # Origin of the map considered in the scenario (in UTM coordinates per now)
    map_tolerance: Optional[int] = 2  # Tolerance for the map simplification process
    map_buffer: Optional[int] = 0  # Buffer for the map simplification process
    ais_data_file: Optional[Path] = None  # Path to the AIS data file, if considered
    ship_data_file: Optional[Path] = None  # Path to the ship information data file associated with AIS data, if considered
    allowed_nav_statuses: Optional[list] = None  # List of AIS navigation statuses that are allowed in the scenario
    n_episodes: Optional[int] = 1  # Number of episodes to run for the scenario. Each episode is a new random realization of the scenario.
    n_random_ships: Optional[int] = None  # Fixed number of random ships in the scenario, excluding the own-ship, if considered
    n_random_ships_range: Optional[list] = None  # Variable range of number of random ships in the scenario, excluding the own-ship, if considered
    ship_list: Optional[list] = None  # List of ship configurations for the scenario, does not have to be equal to the number of ships in the scenario.
    filename: Optional[str] = None  # Filename of the scenario, stored after creation

    def to_dict(self) -> dict:
        output = {
            "name": self.name,
            "save_scenario": self.save_scenario,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "dt_sim": self.dt_sim,
            "type": self.type.name,
            "n_episodes": self.n_episodes,
            "utm_zone": self.utm_zone,
            "map_data_files": self.map_data_files,
            "map_tolerance": self.map_tolerance,
            "map_buffer": self.map_buffer,
            "new_load_of_map_data": self.new_load_of_map_data,
            "ship_list": [],
        }

        if self.n_random_ships is not None:
            output["n_random_ships"] = self.n_random_ships

        if self.n_random_ships_range is not None:
            output["n_random_ships_range"] = self.n_random_ships_range

        if self.map_size is not None:
            output["map_size"] = list(self.map_size)

        if self.map_origin_enu is not None:
            output["map_origin_enu"] = list(self.map_origin_enu)

        if self.ais_data_file is not None:
            output["ais_data_file"] = str(self.ais_data_file)

        if self.ship_data_file is not None:
            output["ship_data_file"] = str(self.ship_data_file)

        if self.allowed_nav_statuses is not None:
            output["allowed_nav_statuses"] = self.allowed_nav_statuses

        if self.filename is not None:
            output["filename"] = self.filename

        if self.ship_list is not None:
            for ship_config in self.ship_list:
                output["ship_list"].append(ship_config.to_dict())

        return output

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = ScenarioConfig(
            name=config_dict["name"],
            save_scenario=config_dict["save_scenario"],
            t_start=config_dict["t_start"],
            t_end=config_dict["t_end"],
            dt_sim=config_dict["dt_sim"],
            type=ScenarioType[config_dict["type"]],
            utm_zone=config_dict["utm_zone"],
            map_data_files=config_dict["map_data_files"],
            new_load_of_map_data=config_dict["new_load_of_map_data"],
            ship_list=[],
        )

        if "n_episodes" in config_dict:
            config.n_episodes = config_dict["n_episodes"]

        if "n_random_ships" in config_dict:
            config.n_random_ships = config_dict["n_random_ships"]

        if "n_random_ships_range" in config_dict:
            config.n_random_ships_range = config_dict["n_random_ships_range"]

        if "map_size" in config_dict:
            config.map_size = tuple(config_dict["map_size"])

        if "map_origin_enu" in config_dict:
            config.map_origin_enu = tuple(config_dict["map_origin_enu"])

        if "map_tolerance" in config_dict:
            config.map_tolerance = config_dict["map_tolerance"]

        if "map_buffer" in config_dict:
            config.map_buffer = config_dict["map_buffer"]

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

        if "filename" in config_dict:
            config.filename = config_dict["filename"]

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

    n_wps_range: list = field(default_factory=lambda: [2, 4])  # Range of number of waypoints to be generated
    speed_plan_variation_range: list = field(default_factory=lambda: [-1.0, 1.0])  # Determines maximal +- change in speed plan from one segment to the next
    waypoint_dist_range: list = field(default_factory=lambda: [200.0, 1000.0])  # Range of [min, max] change in distance between randomly created waypoints
    waypoint_ang_range: list = field(default_factory=lambda: [-45.0, 45.0])  # Range of [min, max] change in angle between randomly created waypoints
    ho_bearing_range: list = field(default_factory=lambda: [-20.0, 20.0])  # Range of [min, max] bearing from the own-ship to the target ship for head-on scenarios
    ho_heading_range: list = field(
        default_factory=lambda: [-15.0, 15.0]
    )  # Range of [min, max] heading variations of the target ship relative to completely reciprocal head-on scenarios
    ot_bearing_range: list = field(default_factory=lambda: [-20.0, 20.0])  # Range of [min, max] bearing from the own-ship to the target ship for overtaking scenarios
    ot_heading_range: list = field(
        default_factory=lambda: [-15.0, 15.0]
    )  # Range of [min, max] heading variations of the target ship relative to completely parallel overtaking scenarios
    cr_bearing_range: list = field(default_factory=lambda: [15.1, 112.5])  # Range of [min, max] bearing from the own-ship to the target ship for crossing scenarios
    cr_heading_range: list = field(
        default_factory=lambda: [-15.0, 15.0]
    )  # Range of [min, max] heading variations of the target ship relative to completely orthogonal crossing scenarios
    dist_between_ships_range: list = field(default_factory=lambda: [200, 10000])  # Range of [min, max] distance variations possible between ships.

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

    def to_dict(self):
        output = asdict(self)
        return output


class ScenarioGenerator:
    """Class for generating maritime traffic scenarios in a given geographical environment.

    Internal variables:
        enc (ENC): Electronic Navigational Chart object containing the geographical environment.
        _config (Config): Configuration object containing all parameters/settings related to the creation of scenarios.
    """

    enc: senc.ENC
    _config: Config

    def __init__(self, config: Optional[Config] = None, enc_config_file: Optional[Path] = dp.seacharts_config, init_enc: bool = False, **kwargs) -> None:
        """Constructor for the ScenarioGenerator.

        Args:
            config (Config): Configuration object containing all parameters/settings related to the creation of scenarios.
            config_file (Path, optional): Absolute path to the generator config file. Defaults to dp.scenario_generator_config.
            **kwargs: Keyword arguments for the ScenarioGenerator, can be e.g.:
                    new_data (bool): Flag determining whether or not to read ENC data from shapefiles again.
        """
        if config:
            self._config = config
        else:
            self._config = Config()

        if init_enc:
            self.enc = senc.ENC(config_file=enc_config_file, **kwargs)

    def _configure_enc(self, scenario_config: ScenarioConfig) -> senc.ENC:
        """Configures the ENC object based on the scenario config file.

        Args:
            scenario_config (ScenarioConfig): Scenario config object.
        """
        # print(f"ENC map size: {scenario_config.map_size}")
        # print(f"ENC map origin: {scenario_config.map_origin_enu}")

        self.enc = senc.ENC(
            config_file=dp.seacharts_config,
            utm_zone=scenario_config.utm_zone,
            size=scenario_config.map_size,
            origin=scenario_config.map_origin_enu,
            files=scenario_config.map_data_files,
            new_data=scenario_config.new_load_of_map_data,
            tolerance=scenario_config.map_tolerance,
            buffer=scenario_config.map_buffer,
        )

        return copy.deepcopy(self.enc)

    def load_scenario_from_folder(self, folder: Path, scenario_name: str, verbose: bool = False) -> list:
        """Loads all episode files for a given scenario from a folder that match the specified `scenario_name`.

        Args:
            folder (Path): Path to folder containing scenario files.
            scenario_name (str): Name of the scenario.

        Returns:
            list: List of scenario files.
        """
        scenario_episode_list = []
        first = True
        for _, file in enumerate(folder.iterdir()):
            if not (scenario_name in file.name and file.suffix == ".yaml"):
                continue

            if verbose:
                print(f"ScenarioGenerator: Loading scenario file: {file.name}...")
            ship_list, config = self.load_episode(config_file=file)
            if first:
                first = False
                enc = self._configure_enc(config)
            scenario_episode_list.append({"ship_list": ship_list, "config": config})
        if verbose:
            print(f"ScenarioGenerator: Finished loading scenario episode files for scenario: {scenario_name}.")
        return (scenario_episode_list, enc)

    def load_episode(self, config_file: Path) -> Tuple[list, ScenarioConfig]:
        """Loads a fully defined scenario episode from configuration file.

        NOTE: The file must have a ship list with fully specified ship configurations,
        and a corresponding correct number of random ships (excluded the own-ship with ID 0).

        NOTE: The scenario ENC object is not initialized here, but in the `load_scenario_from_folder` function.

        Args:
            config_file (Path): Absolute path to the scenario config file.

        Returns:
            Tuple[list, ScenarioConfig]: List of ships in the scenario with initialized poses and plans, the final scenario config object.
        """
        config = cp.extract(ScenarioConfig, config_file, dp.scenario_schema)
        ship_list = []
        for ship_cfg in config.ship_list:
            assert (
                ship_cfg.csog_state is not None and ship_cfg.waypoints is not None and ship_cfg.speed_plan is not None and ship_cfg.id is not None
            ), "A fully specified ship config has an initial csog_state, waypoints, speed_plan and id."
            ship_obj = ship.Ship(mmsi=ship_cfg.mmsi, identifier=ship_cfg.id, config=ship_cfg)
            ship_list.append(ship_obj)

        return ship_list, config

    def generate_scenarios_from_files(self, files: list, verbose: bool = False) -> list:
        """Generates scenarios from each of the input file paths.

        Args:
            files (list): List of scenario files to run, as Path objects.

        Returns:
            list: List of episode config data dictionaries and relevant ENC objects, for each scenario.
        """
        scenario_data_list = []
        for i, scenario_file in enumerate(files):
            if verbose:
                print(f"\rScenario generator: Creating scenario nr {i + 1}: {scenario_file.name}...")
            scenario_episode_list, enc = self.generate(config_file=scenario_file)
            if verbose:
                print(f"\rScenario generator: Finished creating scenario nr {i + 1}: {scenario_file.name}.")
            scenario_data_list.append((scenario_episode_list, enc))
        return scenario_data_list

    def generate(self, config: Optional[ScenarioConfig] = None, config_file: Optional[Path] = None, enc: Optional[senc.ENC] = None) -> Tuple[list, senc.ENC]:
        """Main class function. Creates a maritime scenario, with a number of `n_episodes` based on the input config or config file.

        If specified, the ENC object provides the geographical environment.

        You must provide either a valid scenario config object or a scenario config file path as input.

        Args:
            config (ScenarioConfig, optional): Scenario config object. Defaults to None.
            config_file (Path, optional): Absolute path to the scenario config file. Defaults to None.
            enc (ENC, optional): Electronic Navigational Chart object containing the geographical environment. Defaults to None.

        Returns:
            Tuple[list, ENC]: List of scenario episodes, each containing a dictionary of episode information. Also, the corresponding ENC object is returned.
        """
        if config is None:
            assert config_file is not None, "Either scenario_config or scenario_config_file must be specified."
            config = cp.extract(ScenarioConfig, config_file, dp.scenario_schema)
            config.filename = config_file.name

        ais_vessel_data_list = []
        mmsi_list = []
        ais_data_output = process_ais_data(config)
        if ais_data_output:
            ais_vessel_data_list = ais_data_output["vessels"]
            mmsi_list = ais_data_output["mmsi_list"]
            config.map_origin_enu = ais_data_output["map_origin_enu"]
            config.map_size = ais_data_output["map_size"]

        config.map_origin_enu, config.map_size = find_global_map_origin_and_size(config)
        if enc is not None:
            self.enc = enc
            enc_copy = copy.deepcopy(enc)
        else:
            enc_copy = self._configure_enc(config)

        ais_ship_data = self.generate_ships_with_ais_data(
            ais_vessel_data_list,
            mmsi_list,
            config,
        )

        scenario_episode_list = []
        for ep in range(config.n_episodes):
            episode = {}
            episode["ship_list"], episode["config"] = self.generate_episode(copy.deepcopy(config), ais_ship_data, enc)
            episode["config"].name = f"{config.name}_ep{ep + 1}"
            if config.save_scenario:
                episode["config"].filename = save_scenario_episode_definition(episode["config"])
            scenario_episode_list.append(episode)

        return scenario_episode_list, enc_copy

    def generate_episode(self, config: ScenarioConfig, ais_ship_data: Optional[dict] = None, enc: Optional[senc.ENC] = None) -> Tuple[list, ScenarioConfig]:
        """Creates a single maritime scenario episode based on the input config.

        Some ships in the episode can be partially or fully specified by the AIS ship data, if not none.

        Random plans for each ship will be created unless specified in ship_list entries or loaded from AIS data.

        Args:
            config (ScenarioConfig): Scenario config object.
            ais_ship_data (dict, optional): Dictionary containing AIS ship data. Defaults to None.
            enc (ENC, optional): Electronic Navigational Chart object containing the geographical environment, to override the existing enc being used. Defaults to None.

        Returns:
            Tuple[list, ScenarioConfig]: List of ships in the scenario with initialized poses and plans, the final scenario config object.
        """
        if ais_ship_data is None:
            ship_list = []
            ship_config_list = []
            csog_state_list = []
            non_cfged_ship_indices = []
            cfg_ship_idx = 0
        else:
            ship_list = ais_ship_data["ship_list"].copy()
            ship_config_list = ais_ship_data["ship_config_list"].copy()
            csog_state_list = ais_ship_data["csog_state_list"].copy()
            non_cfged_ship_indices = ais_ship_data["non_cfged_ship_indices"].copy()
            cfg_ship_idx = ais_ship_data["cfg_ship_idx"]

        if enc is not None:
            self.enc = enc

        if config.n_random_ships is not None:
            n_random_ships = config.n_random_ships
        else:
            n_random_ships = random.randint(config.n_random_ships_range[0], config.n_random_ships_range[1])
        config.n_random_ships = n_random_ships

        # Ships still non-configured will be generated randomly
        # Add own-ship (idx 0) if no AIS ships were configured
        n_ais_cfg_ships = len(ship_list)
        if n_ais_cfg_ships == 0 and len(non_cfged_ship_indices) == 0:
            non_cfged_ship_indices.append(0)
            cfg_ship_idx = 1

        n_random_ships += len(non_cfged_ship_indices)
        for i in range(cfg_ship_idx, n_ais_cfg_ships + n_random_ships):
            non_cfged_ship_indices.append(i)

        ship_list, ship_config_list, csog_state_list = self.generate_ships_with_random_plans(non_cfged_ship_indices, ship_list, ship_config_list, csog_state_list, config)
        ship_list.sort(key=lambda x: x.id)
        ship_config_list.sort(key=lambda x: x.id)

        # Overwrite the preliminary ship config list with the final one
        config.ship_list = ship_config_list

        return ship_list, config

    def generate_ships_with_ais_data(
        self,
        ais_vessel_data_list: list,
        mmsi_list: list,
        config: ScenarioConfig,
    ) -> dict:
        """Generates ships from AIS data. Their plans can be fully or partially be specified by the AIS trajectory data.

        Args:
            ais_vessel_data_list (list): List of AIS vessel data objects.
            mmsi_list (list): List of corresponding MMSI numbers for the AIS vessels.
            config (ScenarioConfig): The scenario configuration.

        Returns:
            dict: Dictionary containing the list of AIS ships, the list of AIS ship configurations, the list of AIS CSOG states and the updated list
            of non-configured ship indices. Also, the idx of the next ship to be configured (if any) is stored.
        """
        output = {}
        cfg_ship_idx = 0
        non_cfged_ship_indices = []
        ship_list = []
        ship_config_list = []
        csog_state_list = []
        n_cfg_ships = len(config.ship_list)
        while ais_vessel_data_list:
            use_ais_ship_trajectory = True
            if cfg_ship_idx < n_cfg_ships and cfg_ship_idx == config.ship_list[cfg_ship_idx].id:
                ship_config = config.ship_list[cfg_ship_idx]
            else:
                ship_config = ship.Config()
                ship_config.id = cfg_ship_idx
                ship_config.mmsi = cfg_ship_idx + 1

            if ship_config.random_generated:
                non_cfged_ship_indices.append(cfg_ship_idx)
                cfg_ship_idx += 1
                continue

            ship_obj = ship.Ship(mmsi=cfg_ship_idx + 1, identifier=cfg_ship_idx, config=ship_config)

            # The own-ship (with index 0) will not use the predefined AIS trajectory.
            idx = 0
            if cfg_ship_idx == 0:
                use_ais_ship_trajectory = False

            if ship_config.mmsi in mmsi_list:
                # use_ais_ship_trajectory = False
                idx = [i for i in range(len(ais_vessel_data_list)) if ais_vessel_data_list[i].mmsi == ship_config.mmsi][0]

            ais_vessel = ais_vessel_data_list.pop(idx)
            if ais_vessel.status.value not in config.allowed_nav_statuses:
                continue

            ship_obj.transfer_vessel_ais_data(ais_vessel, use_ais_ship_trajectory, ship_config.t_start, ship_config.t_end)
            ship_config.mmsi = ship_obj.mmsi

            if not use_ais_ship_trajectory and ship_config.waypoints is None:
                waypoints = self.generate_random_waypoints(ship_obj.csog_state[0], ship_obj.csog_state[1], ship_obj.csog_state[3], ship_obj.draft)
                speed_plan = self.generate_random_speed_plan(ship_obj.csog_state[2], U_min=ship_obj.min_speed, U_max=ship_obj.max_speed, n_wps=waypoints.shape[1])
                ship_config.waypoints = waypoints
                ship_config.speed_plan = speed_plan
                ship_obj.set_nominal_plan(waypoints, speed_plan)

            csog_state_list.append(ship_obj.csog_state)
            ship_list.append(ship_obj)
            ship_config_list.append(ship_config)
            cfg_ship_idx += 1

        output["ship_list"] = ship_list
        output["ship_config_list"] = ship_config_list
        output["csog_state_list"] = csog_state_list
        output["non_cfged_ship_indices"] = non_cfged_ship_indices
        output["cfg_ship_idx"] = cfg_ship_idx
        return output

    def generate_ships_with_random_plans(
        self,
        non_cfged_ship_indices: list,
        ship_list: list,
        ship_config_list: list,
        csog_state_list: list,
        config: ScenarioConfig,
    ) -> Tuple[list, list, list]:
        """Generates ships with random plans.

        Args:
            non_cfged_ship_indices (list): List of indices of ships that are not yet configured.
            ship_list (list): List of already configured ships, to which the random ships will be added.
            ship_config_list (list): List of final ship configurations, to which the random ships will be added.
            csog_state_list (list): List of CSOG states of the already configured ships, to which the random ship initial CSOG states will be added.
            config (ScenarioConfig): The scenario configuration.

        Returns:
            Tuple[list, list, list]: The list of ships, the list of ship configurations, and the list of CSOG states.
        """
        # Number of ships that are configured for the scenario
        n_cfg_ships = len(config.ship_list)
        os_csog_state = [x.csog_state for x in ship_list if x.id == 0]
        while non_cfged_ship_indices:
            cfg_ship_idx = non_cfged_ship_indices.pop(0)
            if cfg_ship_idx < n_cfg_ships and cfg_ship_idx == config.ship_list[cfg_ship_idx].id:
                ship_config = config.ship_list[cfg_ship_idx]
            else:
                ship_config = ship.Config()
                ship_config.id = cfg_ship_idx
                ship_config.mmsi = cfg_ship_idx + 1

            ship_obj = ship.Ship(mmsi=cfg_ship_idx + 1, identifier=cfg_ship_idx, config=ship_config)

            # Target ship poses are created relative to the own-ship (idx 0).
            csog_state = ship_config.csog_state
            if ship_config.csog_state is None:
                if cfg_ship_idx == 0:
                    csog_state = self.generate_random_csog_state(U_min=5.0, U_max=ship_obj.max_speed, draft=ship_obj.draft, min_land_clearance=ship_obj.length * 2.0)
                else:
                    csog_state = self.generate_ts_csog_state(
                        config.type,
                        os_csog_state,
                        U_min=ship_obj.min_speed,
                        U_max=ship_obj.max_speed,
                        draft=ship_obj.draft,
                        min_land_clearance=ship_obj.length * 3.0,
                    )
                ship_config.csog_state = csog_state
                ship_obj.set_initial_state(csog_state)

            if cfg_ship_idx == 0:
                os_csog_state = csog_state

            if ship_config.waypoints is None:
                waypoints = self.generate_random_waypoints(csog_state[0], csog_state[1], csog_state[3], ship_obj.draft)
                speed_plan = self.generate_random_speed_plan(csog_state[2], U_min=ship_obj.min_speed, U_max=ship_obj.max_speed, n_wps=waypoints.shape[1])
                ship_config.waypoints = waypoints
                ship_config.speed_plan = speed_plan
                ship_obj.set_nominal_plan(waypoints, speed_plan)

            ship_config.random_generated = True

            csog_state_list.append(csog_state)
            ship_list.append(ship_obj)
            ship_config_list.append(ship_config)

        return ship_list, ship_config_list, csog_state_list

    def generate_ts_csog_state(
        self,
        scenario_type: ScenarioType,
        os_csog_state: np.ndarray,
        U_min: float = 1.0,
        U_max: float = 15.0,
        draft: float = 2.0,
        min_land_clearance: float = 10.0,
    ) -> np.ndarray:
        """Generates a position for the target ship based on the perspective of the first ship/own-ship,
        such that the scenario is of the input type.

        Args:
            scenario_type (ScenarioType): Type of scenario.
            os_csog_state (np.ndarray): Own-ship COG-SOG state = [x, y, speed, heading].
            U_min (float, optional): Obstacle minimum speed. Defaults to 1.0.
            U_max (float, optional): Obstacle maximum speed. Defaults to 15.0.
            draft (float, optional): Draft of target ship. Defaults to 2.0.
            min_land_clearance (float, optional): Minimum distance between target ship and land. Defaults to 100.0.

        Returns:
            np.ndarray: Target ship position = [x, y].
        """

        if any(np.isnan(os_csog_state)):
            return self.generate_random_csog_state(U_min=U_min, U_max=U_max, draft=draft, min_land_clearance=min_land_clearance)

        if scenario_type == ScenarioType.MS:
            scenario_type = random.choice([ScenarioType.HO, ScenarioType.OT_ing, ScenarioType.OT_en, ScenarioType.CR_GW, ScenarioType.CR_SO])

        if scenario_type == ScenarioType.OT_en and U_max - 2.0 <= os_csog_state[2]:
            print(
                "WARNING: ScenarioType = OT_en: Own-ship speed should be below the maximum target ship speed minus margin of 2.0. Selecting a different scenario type..."
            )
            scenario_type = random.choice([ScenarioType.HO, ScenarioType.OT_ing, ScenarioType.CR_GW, ScenarioType.CR_SO])

        if scenario_type == ScenarioType.OT_ing and U_min >= os_csog_state[2] - 2.0:
            print(
                "WARNING: ScenarioType = OT_ing: Own-ship speed minus margin of 2.0 should be above the minimum target ship speed. Selecting a different scenario type..."
            )
            scenario_type = random.choice([ScenarioType.HO, ScenarioType.OT_en, ScenarioType.CR_GW, ScenarioType.CR_SO])

        is_safe_pose = False
        iter_count = 1
        while not is_safe_pose:
            if scenario_type == ScenarioType.HO:
                bearing = random.uniform(self._config.ho_bearing_range[0], self._config.ho_bearing_range[1])
                speed = random.uniform(U_min, U_max)
                heading_modifier = 180.0 + random.uniform(self._config.ho_heading_range[0], self._config.ho_heading_range[1])

            elif scenario_type == ScenarioType.OT_ing:
                bearing = random.uniform(self._config.ot_bearing_range[0], self._config.ot_bearing_range[1])
                speed = random.uniform(U_min, os_csog_state[2] - 2.0)
                heading_modifier = random.uniform(self._config.ot_heading_range[0], self._config.ot_heading_range[1])

            elif scenario_type == ScenarioType.OT_en:
                bearing = random.uniform(self._config.ot_bearing_range[0], self._config.ot_bearing_range[1])
                speed = random.uniform(os_csog_state[2], U_max)
                heading_modifier = random.uniform(self._config.ot_heading_range[0], self._config.ot_heading_range[1])

            elif scenario_type == ScenarioType.CR_GW:
                bearing = random.uniform(self._config.cr_bearing_range[0], self._config.cr_bearing_range[1])
                speed = random.uniform(U_min, U_max)
                heading_modifier = -90.0 + random.uniform(self._config.cr_heading_range[0], self._config.cr_heading_range[1])

            elif scenario_type == ScenarioType.CR_SO:
                bearing = random.uniform(-self._config.cr_bearing_range[1], -self._config.cr_bearing_range[0])
                speed = random.uniform(U_min, U_max)
                heading_modifier = 90.0 + random.uniform(self._config.cr_heading_range[0], self._config.cr_heading_range[1])

            else:
                bearing = random.uniform(0.0, 2.0 * np.pi)
                speed = random.uniform(U_min, U_max)
                heading_modifier = random.uniform(0.0, 359.999)

            bearing = np.deg2rad(bearing)
            heading = os_csog_state[3] + np.deg2rad(heading_modifier)

            distance_os_ts = random.uniform(self._config.dist_between_ships_range[0], self._config.dist_between_ships_range[1])
            x = os_csog_state[0] + distance_os_ts * np.cos(os_csog_state[3] + bearing)
            y = os_csog_state[1] + distance_os_ts * np.sin(os_csog_state[3] + bearing)

            distance_to_land = mapf.min_distance_to_land(self.enc, y, x)

            if distance_to_land >= min_land_clearance:
                is_safe_pose = True

            iter_count += 1
            if iter_count >= 100000:
                raise ValueError("Could not find a safe COG-SOG state for the target ship. Have you remembered new load of map data?")

        return np.array([x, y, speed, heading])

    def generate_random_csog_state(
        self,
        U_min: float = 1.0,
        U_max: float = 15.0,
        draft: float = 3.0,
        heading: Optional[float] = None,
        min_land_clearance: float = 100.0,
    ) -> np.ndarray:
        """Creates a random COG-SOG state which adheres to the ship's draft and maximum speed.

        Args:
            U_min (float, optional): Minimum speed of the ship. Defaults to 1.0.
            U_max (float, optional): Maximum speed of the ship. Defaults to 15.0.
            draft (float, optional): How deep the ship keel is into the water. Defaults to 5.
            heading (Optional[float], optional): Heading of the ship in radians. Defaults to None.
            min_land_clearance (float, optional): Minimum distance to land. Defaults to 100.0.

        Returns:
            np.ndarray: Array containing the vessel state = [x, y, speed, heading]
        """
        x, y = mapf.generate_random_start_position_from_draft(self.enc, draft, min_land_clearance)
        speed = random.uniform(U_min, U_max)
        if heading is None:
            heading = random.uniform(0.0, 2.0 * np.pi)

        return np.array([x, y, speed, heading])

    def generate_random_waypoints(self, x: float, y: float, psi: float, draft: float = 5.0, n_wps: Optional[int] = None) -> np.ndarray:
        """Creates random waypoints starting from a ship position and heading.

        Args:
            x (float): x position (north) of the ship.
            y (float): y position (east) of the ship.
            psi (float): heading of the ship in radians.
            draft (float, optional): How deep the ship keel is into the water. Defaults to 5.
            n_wps (Optional[int]): Number of waypoints to create.

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
            iter_count = -1
            while crosses_grounding_hazards:
                iter_count += 1

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

                if iter_count >= 20:
                    break

            if iter_count >= 20:
                waypoints = waypoints[:, 0:i]
                if i == 1:  # stand-still, no waypoints under given parameters that avoids grounding hazards
                    waypoints = np.append(waypoints, waypoints, axis=1)
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


def save_scenario_episode_definition(scenario_config: ScenarioConfig) -> str:
    """Saves the the scenario episode defined by the preliminary scenario configuration and list of configured ships.

    Uses the config to create a unique scenario name and filename. The scenario is saved in the default scenario save folder.

    Args:
        scenario_config (ScenarioConfig): Scenario configuration

    Returns:
        str: The filename of the saved scenario.
    """
    scenario_config_dict: dict = scenario_config.to_dict()
    scenario_config_dict["save_scenario"] = False
    if "n_episodes" in scenario_config_dict:
        scenario_config_dict.pop("n_episodes")  # Do not save the number of episodes for the single scenario episode
    if "n_random_ships_range" in scenario_config_dict:
        scenario_config_dict.pop("n_random_ships_range")  # Do not save the n_random_ships_range for the single scenario episode
    current_datetime_str = mhm.current_utc_datetime_str("%d%m%Y_%H%M%S")
    scenario_config_dict["name"] = scenario_config_dict["name"] + "_" + current_datetime_str
    filename = scenario_config.name + "_" + current_datetime_str + ".yaml"
    scenario_config_dict["filename"] = filename
    save_file = dp.saved_scenarios / filename
    with save_file.open(mode="w") as file:
        yaml.dump(scenario_config_dict, file)

    return filename


def find_global_map_origin_and_size(config: ScenarioConfig) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Finds the global map origin and size encompassing all ships in the scenario.

    Args:
        config (ScenarioConfig): Scenario configuration

    Returns:
        Tuple[np.ndarray, np.ndarray]: Global map origin and size
    """
    assert config.map_origin_enu is not None and config.map_size is not None
    assert config.ship_list is not None
    map_origin_enu = config.map_origin_enu
    map_size = config.map_size

    min_east = map_origin_enu[0]
    min_north = map_origin_enu[1]
    max_east = map_origin_enu[0] + map_size[0]
    max_north = map_origin_enu[1] + map_size[1]
    for ship_config in config.ship_list:
        if ship_config.csog_state is not None:
            csog_state = ship_config.csog_state
            if csog_state[0] < min_north:
                min_north = csog_state[0]
            if csog_state[0] > max_north:
                max_north = csog_state[0]
            if csog_state[1] < min_east:
                min_east = csog_state[1]
            if csog_state[1] > max_east:
                max_east = csog_state[1]

        if ship_config.waypoints is not None:
            waypoints = ship_config.waypoints
            if np.min(waypoints[0, :]) < min_north:
                min_north = np.min(waypoints[0, :])
            if np.max(waypoints[0, :]) > max_north:
                max_north = np.max(waypoints[0, :])
            if np.min(waypoints[1, :]) < min_east:
                min_east = np.min(waypoints[1, :])
            if np.max(waypoints[1, :]) > max_east:
                max_east = np.max(waypoints[1, :])

    map_origin_enu = min_east, min_north
    map_size = max_east - min_east, max_north - min_north
    return map_origin_enu, map_size


def process_ais_data(config: ScenarioConfig) -> dict:
    """Processes AIS data from a list of AIS data files, returns a dict containing AIS VesselData, ship MMSIs and the coordinate frame origin and size.

    Args:
        config (ScenarioConfig): Configuration object containing all parameters/settings related to the creation of a scenario.

    Returns:
        dict: Dict containing AIS VesselData, ship MMSIs, and the coordinate frame origin and size.
    """
    output = {}
    if config.ais_data_file is not None:
        output = colav_eval_fu.read_ais_data(config.ais_data_file, config.ship_data_file, config.utm_zone, config.map_origin_enu, config.map_size, config.dt_sim)
    return output
