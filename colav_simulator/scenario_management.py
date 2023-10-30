"""
    scenario_management.py

    Summary:
        Contains functionality for loading existing scenario definitions,
        and also a ScenarioGenerator class for generating new scenarios. Functionality
        for saving these new scenarios also exists.

    Author: Trym Tengesdal, Joachim Miller, Melih Akdag
"""

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import colav_simulator.behavior_generator as bg
import colav_simulator.common.config_parsing as cp
import colav_simulator.common.file_utils as file_utils
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp
import colav_simulator.core.ship as ship
import colav_simulator.core.stochasticity as stoch
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
    """Configuration class for a maritime COLAV scenario."""

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
    map_tolerance: Optional[int] = 0  # Tolerance for the map simplification process
    map_buffer: Optional[int] = 0  # Buffer for the map simplification process
    ais_data_file: Optional[Path] = None  # Path to the AIS data file, if considered
    ship_data_file: Optional[Path] = None  # Path to the ship information data file associated with AIS data, if considered
    allowed_nav_statuses: Optional[list] = None  # List of AIS navigation statuses that are allowed in the scenario
    n_episodes: Optional[int] = 1  # Number of episodes to run for the scenario. Each episode is a new random realization of the scenario.
    n_random_ships: Optional[int] = None  # Fixed number of random ships in the scenario, excluding the own-ship, if considered
    n_random_ships_range: Optional[list] = None  # Variable range of number of random ships in the scenario, excluding the own-ship, if considered
    ship_list: Optional[list] = None  # List of ship configurations for the scenario, does not have to be equal to the number of ships in the scenario.
    filename: Optional[str] = None  # Filename of the scenario, stored after creation
    stochasticity: Optional[stoch.Config] = None  # Configuration class containing stochasticity parameters for the scenario
    rl_observation_type: Optional[dict] = field(
        default_factory=lambda: {"tuple_observation": ["navigation_state_observation", "lidar_like_observation"]}
    )  # Observation type settings configured for an  RL agent
    rl_action_type: Optional[str] = "continuous_autopilot_reference_action"  # Observation type configured for an  RL agent

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
            "stochasticity": self.stochasticity,
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

        if self.stochasticity is not None:
            output["stochasticity"] = self.stochasticity.to_dict()

        if self.ship_list is not None:
            for ship_config in self.ship_list:
                output["ship_list"].append(ship_config.to_dict())

        if self.rl_observation_type is not None:
            output["rl_observation_type"] = self.rl_observation_type

        if self.rl_action_type is not None:
            output["rl_action_type"] = self.rl_action_type

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

        if "stochasticity" in config_dict:
            config.stochasticity = stoch.Config.from_dict(config_dict["stochasticity"])

        if "ship_list" in config_dict:
            config.ship_list = []
            for ship_config in config_dict["ship_list"]:
                config.ship_list.append(ship.Config.from_dict(ship_config))

        if "rl_observation_type" in config_dict:
            config.rl_observation_type = config_dict["rl_observation_type"]

        if "rl_action_type" in config_dict:
            config.rl_action_type = config_dict["rl_action_type"]

        return config


@dataclass
class Config:
    """Configuration class for managing all parameters/settings related to the creation of scenarios.
    All angle ranges are in degrees, and all distances are in meters.
    """

    verbose: bool = False
    behavior_generator: bg.Config = field(default_factory=lambda: bg.Config())
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
    scenario_files: Optional[list] = None
    scenario_folder: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config()
        if "scenario_files" in config_dict:
            config.scenario_files = config_dict["scenario_files"]

        if "scenario_folder" in config_dict:
            config.scenario_folder = config_dict["scenario_folder"]
            config.scenario_files = None

        config.behavior_generator = bg.Config.from_dict(config_dict["behavior_generator"])
        return config

    def to_dict(self):
        output = asdict(self)
        output.behavior_generator = self.behavior_generator.to_dict()
        return output


class ScenarioGenerator:
    """Class for generating maritime traffic scenarios in a given geographical environment."""

    rng: np.random.Generator
    enc: senc.ENC
    behavior_generator: bg.BehaviorGenerator
    safe_sea_cdt: Optional[list] = None
    rrts: Optional[list] = None
    _config: Config

    def __init__(
        self,
        config: Optional[Config] = None,
        config_file: Optional[Path] = dp.scenario_generator_config,
        enc_config_file: Optional[Path] = dp.seacharts_config,
        init_enc: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Constructor for the ScenarioGenerator.

        Args:
            - config (Config): Configuration object containing all parameters/settings related to the creation of scenarios.
            - config_file (Path, optional): Absolute path to the generator config file. Defaults to dp.scenario_generator_config.
            - enc_config_file (Path, optional): Absolute path to the ENC config file. Defaults to dp.seacharts_config.
            - init_enc (bool, optional): Flag determining whether or not to initialize the ENC object. Defaults to False.
            - seed (Optional[int], optional): Integer seed. Defaults to None.
            - **kwargs: Keyword arguments for the ScenarioGenerator, can be e.g.:
                    new_data (bool): Flag determining whether or not to read ENC data from shapefiles again.
        """
        self._config = Config()
        if config:
            self._config = config
        elif config_file:
            self._config = cp.extract(Config, config_file, dp.scenario_generator_schema)

        self.safe_sea_cdt = None
        if init_enc:
            self.enc = senc.ENC(config_file=enc_config_file, **kwargs)

        self.rng = np.random.default_rng(seed=seed)
        self.behavior_generator = bg.BehaviorGenerator(self._config.behavior_generator)

    def seed(self, seed: Optional[int] = None) -> None:
        """Seeds the random number generator.

        Args:
            seed (Optional[int]): Integer seed. Defaults to None.
        """
        self.rng = np.random.default_rng(seed=seed)

    def _configure_enc(self, scenario_config: ScenarioConfig) -> senc.ENC:
        """Configures the ENC object based on the scenario config file.

        Args:
            - scenario_config (ScenarioConfig): Scenario config object.

        Returns:
            - (senc.ENC): Configured ENC object.
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

    def create_file_path_list_from_config(self) -> list:
        """Creates a list of file paths from the config file scenario files or scenario folder.

        Returns:
            list: List of valid file paths.
        """
        if self._config.scenario_files is not None:
            return [dp.scenarios / f for f in self._config.scenario_files]
        else:
            scenario_folder = dp.scenarios / self._config.scenario_folder
            files = [file for file in scenario_folder.iterdir()]
            files.sort()
            return files

    def generate_configured_scenarios(self) -> list:
        """Generates the list of configured scenarios from the class config file.

        Returns:
            list: List of fully configured scenario data definitions.
        """
        files = self.create_file_path_list_from_config()
        scenario_data_list = self.generate_scenarios_from_files(files)
        return scenario_data_list

    def load_scenario_from_folder(self, folder: Path, scenario_name: str) -> Tuple[list, senc.ENC]:
        """Loads all episode files for a given scenario from a folder that match the specified `scenario_name`.

        Args:
            - folder (Path): Path to folder containing scenario files.
            - scenario_name (str): Name of the scenario.

        Returns:
            - Tuple[list, senc.ENC]: List of scenario files and the corresponding ENC object.
        """
        scenario_episode_list = []
        first = True
        for _, file in enumerate(folder.iterdir()):
            if not (scenario_name in file.name and file.suffix == ".yaml"):
                continue

            if self._config.verbose:
                print(f"ScenarioGenerator: Loading scenario file: {file.name}...")
            ship_list, config = self.load_episode(config_file=file)
            if first:
                first = False
                enc = self._configure_enc(config)
            scenario_episode_list.append({"ship_list": ship_list, "config": config})
        if self._config.verbose:
            print(f"ScenarioGenerator: Finished loading scenario episode files for scenario: {scenario_name}.")
        return (scenario_episode_list, enc)

    def load_episode(self, config_file: Path) -> Tuple[list, ScenarioConfig]:
        """Loads a fully defined scenario episode from configuration file.

        NOTE: The file must have a ship list with fully specified ship configurations,
        and a corresponding correct number of random ships (excluded the own-ship with ID 0).

        NOTE: The scenario ENC object is not initialized here, but in the `load_scenario_from_folder` function.

        Args:
            - config_file (Path): Absolute path to the scenario config file.

        Returns:
            - Tuple[list, ScenarioConfig]: List of ships in the scenario with initialized poses and plans, the final scenario config object.
        """
        config = cp.extract(ScenarioConfig, config_file, dp.scenario_schema)
        ship_list = []
        for ship_cfg in config.ship_list:
            assert (
                ship_cfg.csog_state is not None
                and ((ship_cfg.waypoints is not None and ship_cfg.speed_plan) or ship_cfg.goal_csog_state) is not None
                and ship_cfg.id is not None
            ), "A fully specified ship config has an id, initial csog_state, waypoints + speed_plan or goal state."
            ship_obj = ship.Ship(mmsi=ship_cfg.mmsi, identifier=ship_cfg.id, config=ship_cfg)
            ship_list.append(ship_obj)

        return ship_list, config

    def generate_scenarios_from_files(self, files: list) -> list:
        """Generates scenarios from each of the input file paths.

        Args:
            - files (list): List of configuration files to generate scenarios from, as Path objects.

        Returns:
            - list: List of episode config data dictionaries and relevant ENC objects, for each scenario.
        """
        scenario_data_list = []
        for i, scenario_file in enumerate(files):
            if self._config.verbose:
                print(f"\rScenario generator: Creating scenario nr {i + 1}: {scenario_file.name}...")
            scenario_episode_list, enc = self.generate(config_file=scenario_file)
            if self._config.verbose:
                print(f"\rScenario generator: Finished creating scenario nr {i + 1}: {scenario_file.name}.")
            scenario_data_list.append((scenario_episode_list, enc))
        return scenario_data_list

    def generate(
        self, config: Optional[ScenarioConfig] = None, config_file: Optional[Path] = None, enc: Optional[senc.ENC] = None, new_load_of_map_data: Optional[bool] = None
    ) -> Tuple[list, senc.ENC]:
        """Main class function. Creates a maritime scenario, with a number of `n_episodes` based on the input config or config file.

        If specified, the ENC object provides the geographical environment.

        Args:
            - config (ScenarioConfig, optional): Scenario config object. Defaults to None.
            - config_file (Path, optional): Absolute path to the scenario config file. Defaults to None.
            - enc (ENC, optional): Electronic Navigational Chart object containing the geographical environment. Defaults to None.
            - new_load_of_map_data (bool, optional): Flag determining whether or not to read ENC data from shapefiles again. Defaults to True.

        Returns:
            - Tuple[list, ENC]: List of scenario episodes, each containing a dictionary of episode information. Also, the corresponding ENC object is returned.
        """
        if config is None and config_file is not None:
            config = cp.extract(ScenarioConfig, config_file, dp.scenario_schema)
            config.filename = config_file.name

        if config is None and config_file is None:
            config = cp.extract(ScenarioConfig, self.create_file_path_list_from_config()[0], dp.scenario_schema)

        assert config is not None, "config should not be none here."
        ais_vessel_data_list = []
        mmsi_list = []
        ais_data_output = process_ais_data(config)
        if ais_data_output:
            ais_vessel_data_list = ais_data_output["vessels"]
            mmsi_list = ais_data_output["mmsi_list"]
            config.map_origin_enu = ais_data_output["map_origin_enu"]
            config.map_size = ais_data_output["map_size"]
        config.map_origin_enu, config.map_size = find_global_map_origin_and_size(config)
        if new_load_of_map_data is not None:
            config.new_load_of_map_data = new_load_of_map_data

        if enc is not None:
            self.enc = enc
            enc_copy = copy.deepcopy(enc)
        else:
            enc_copy = self._configure_enc(config)

        if self.safe_sea_cdt is None:
            self.safe_sea_cdt = mapf.create_safe_sea_triangulation(self.enc, vessel_min_depth=2, show_plots=False)

        if config.n_random_ships is not None:
            n_random_ships = config.n_random_ships
        else:
            n_random_ships = self.rng.integers(config.n_random_ships_range[0], config.n_random_ships_range[1])
        config.n_random_ships = n_random_ships

        # Create partially defined ship objects and ship configurations for all ships
        ship_list = []
        ship_config_list = []
        n_cfg_ships = len(config.ship_list)
        for s in range(1 + config.n_random_ships):  # +1 for own-ship
            if s < n_cfg_ships and s == config.ship_list[s].id:
                ship_config = config.ship_list[s]
            else:
                ship_config = ship.Config()
                ship_config.id = s
                ship_config.mmsi = s + 1

            ship_obj = ship.Ship(mmsi=ship_config.mmsi, identifier=ship_config.id, config=ship_config)
            ship_list.append(ship_obj)
            ship_config_list.append(ship_config)

        config.ship_list = ship_config_list

        scenario_episode_list = []
        for ep in range(config.n_episodes):
            episode = {}
            episode["ship_list"], episode["disturbance"], episode["config"] = self.generate_episode(
                copy.deepcopy(ship_list), copy.deepcopy(config), ais_vessel_data_list, mmsi_list, enc
            )
            episode["config"].name = f"{config.name}_ep{ep + 1}"
            if config.save_scenario:
                episode["config"].filename = save_scenario_episode_definition(episode["config"])
            scenario_episode_list.append(episode)

        return scenario_episode_list, enc_copy

    def generate_episode(
        self, ship_list: list, config: ScenarioConfig, ais_vessel_data_list: Optional[list], mmsi_list: Optional[list], enc: Optional[senc.ENC] = None
    ) -> Tuple[list, Optional[stoch.Disturbance], ScenarioConfig]:
        """Creates a single maritime scenario episode.

        Some ships in the episode can be partially or fully specified by the AIS ship data, if not none.

        Random plans for each ship will be created unless specified in ship_list entries or loaded from AIS data.

        Args:
            - ship_list (list): List of ships to be considered in simulation.
            - config (ScenarioConfig): Scenario config object.
            - ais_vessel_data_list (Optional[list]): Optional list of AIS vessel data objects.
            - mmsi_list (Optional[list]): Optional list of corresponding MMSI numbers for the AIS vessels.
            - enc (Optional[ENC]): Electronic Navigational Chart object containing the geographical environment, to override the existing enc being used. Defaults to None.

        Returns:
            - Tuple[list, Optional[stoch.Disturbance], ScenarioConfig]: List of ships in the scenario with initialized poses and plans, the disturbance object for the episode (if specified) and the final scenario config object.
        """
        if enc is not None:
            self.enc = enc

        ship_list, config = self.transfer_vessel_ais_data(ship_list, config, ais_vessel_data_list, mmsi_list)

        ship_list, config, csog_state_list = self.generate_ship_csog_states(ship_list, config, self.enc)

        self.behavior_generator.setup(self.rng, ship_list, self.enc, config.t_end - config.t_start)
        ship_list, config.ship_list = self.behavior_generator.generate(self.rng, ship_list, config.ship_list)

        ship_list.sort(key=lambda x: x.id)
        config.ship_list.sort(key=lambda x: x.id)

        disturbance = self.generate_disturbance(config)
        return ship_list, disturbance, config

    def transfer_vessel_ais_data(
        self, ship_list: list, config: ScenarioConfig, ais_vessel_data_list: Optional[list], mmsi_list: Optional[list]
    ) -> Tuple[list, ScenarioConfig]:
        """Transfers AIS vessel data to the ship objects and ship configurations, if available.

        Args:
            - ship_list (list): List of ships to be considered in simulation.
            - config (ScenarioConfig): Scenario config object.
            - ais_vessel_data_list (Optional[list]): Optional list of AIS vessel data objects.
            - mmsi_list (Optional[list]): Optional list of corresponding MMSI numbers for the AIS vessels.

        Returns:
            - Tuple[list, ScenarioConfig]: List of partially initialized ships in the scenario, and the corresponding updated scenario config object.
        """
        if ais_vessel_data_list is None or mmsi_list is None:
            return ship_list, config

        for ship_cfg_idx, ship_config in enumerate(config.ship_list):
            use_ais_ship_trajectory = True
            if ship_config.random_generated:
                continue

            # The own-ship (with index 0) will not use the predefined AIS trajectory.
            idx = 0
            if ship_cfg_idx == 0:
                use_ais_ship_trajectory = False

            if ship_config.mmsi in mmsi_list:
                idx = [i for i in range(len(ais_vessel_data_list)) if ais_vessel_data_list[i].mmsi == ship_config.mmsi][0]

            ais_vessel = ais_vessel_data_list.pop(idx)
            while ais_vessel.status.value not in config.allowed_nav_statuses:
                ais_vessel = ais_vessel_data_list.pop(idx)

            ship_list[ship_cfg_idx].transfer_vessel_ais_data(ais_vessel, use_ais_ship_trajectory, ship_config.t_start, ship_config.t_end)
            ship_config.csog_state = ship_list[ship_cfg_idx].csog_state
            ship_config.mmsi = ship_list[ship_cfg_idx].mmsi

        return ship_list, config

    def generate_disturbance(self, config: ScenarioConfig) -> Optional[stoch.Disturbance]:
        """Generates a disturbance object from the scenario config.

        Args:
            - config (ScenarioConfig): Scenario config object.

        Returns:
            - stoch.Disturbance: Disturbance object.
        """
        if config.stochasticity is None:
            return None

        return stoch.Disturbance(config.stochasticity)

    def generate_ship_csog_states(self, ship_list: list, config: ScenarioConfig) -> Tuple[list, ScenarioConfig, list]:
        """Generates the initial ship poses for the scenario episode.

        Args:
            ship_list (list): List of ships to be considered in simulation.
            config (ScenarioConfig): Scenario config object.

        Returns:
            Tuple[list, ScenarioConfig, list]: List of partially initialized ships in the scenario with poses set, the updated scenario config object and list of generated/set csog states.
        """
        csog_state_list = []
        for ship_cfg_idx, ship_config in enumerate(config.ship_list):
            if ship_config.csog_state is not None:
                continue

            ship_obj = ship_list[ship_cfg_idx]
            if ship_cfg_idx == 0:
                csog_state = self.generate_random_csog_state(U_min=0.0, U_max=ship_obj.max_speed, draft=ship_obj.draft, min_land_clearance=ship_obj.length * 3.0)
            else:
                csog_state = self.generate_target_ship_csog_state(
                    config.type,
                    csog_state_list[0],
                    U_min=ship_obj.min_speed,
                    U_max=ship_obj.max_speed,
                    draft=ship_obj.draft,
                    min_land_clearance=ship_obj.length * 3.0,
                )
            ship_config.csog_state = csog_state
            ship_obj.set_initial_state(ship_config.csog_state)
            csog_state_list.append(ship_config.csog_state)

        return ship_list, config, csog_state_list

    def generate_target_ship_csog_state(
        self,
        scenario_type: ScenarioType,
        os_csog_state: np.ndarray,
        U_min: float = 0.0,
        U_max: float = 15.0,
        draft: float = 2.0,
        min_land_clearance: float = 50.0,
    ) -> np.ndarray:
        """Generates a position for the target ship based on the perspective of the first ship/own-ship,
        such that the scenario is of the input type.

        Args:
            - scenario_type (ScenarioType): Type of scenario.
            - os_csog_state (np.ndarray): Own-ship COG-SOG state = [x, y, speed, heading].
            - U_min (float, optional): Obstacle minimum speed. Defaults to 1.0.
            - U_max (float, optional): Obstacle maximum speed. Defaults to 15.0.
            - draft (float, optional): Draft of target ship. Defaults to 2.0.
            - min_land_clearance (float, optional): Minimum distance between target ship and land. Defaults to 100.0.

        Returns:
            - np.ndarray: Target ship position = [x, y].
        """

        if any(np.isnan(os_csog_state)):
            return self.generate_random_csog_state(U_min=U_min, U_max=U_max, draft=draft, min_land_clearance=min_land_clearance)

        if scenario_type == ScenarioType.MS:
            scenario_type = self.rng.choice([ScenarioType.HO, ScenarioType.OT_ing, ScenarioType.OT_en, ScenarioType.CR_GW, ScenarioType.CR_SO])

        if scenario_type == ScenarioType.OT_en and U_max - 2.0 <= os_csog_state[2]:
            print(
                "WARNING: ScenarioType = OT_en: Own-ship speed should be below the maximum target ship speed minus margin of 2.0. Selecting a different scenario type..."
            )
            scenario_type = self.rng.choice([ScenarioType.HO, ScenarioType.OT_ing, ScenarioType.CR_GW, ScenarioType.CR_SO])

        if scenario_type == ScenarioType.OT_ing and U_min >= os_csog_state[2] - 2.0:
            print(
                "WARNING: ScenarioType = OT_ing: Own-ship speed minus margin of 2.0 should be above the minimum target ship speed. Selecting a different scenario type..."
            )
            scenario_type = self.rng.choice([ScenarioType.HO, ScenarioType.OT_en, ScenarioType.CR_GW, ScenarioType.CR_SO])

        is_safe_pose = False
        iter_count = 1
        while not is_safe_pose:
            if scenario_type == ScenarioType.HO:
                bearing = self.rng.uniform(self._config.ho_bearing_range[0], self._config.ho_bearing_range[1])
                speed = self.rng.uniform(U_min, U_max)
                heading_modifier = 180.0 + self.rng.uniform(self._config.ho_heading_range[0], self._config.ho_heading_range[1])

            elif scenario_type == ScenarioType.OT_ing:
                bearing = self.rng.uniform(self._config.ot_bearing_range[0], self._config.ot_bearing_range[1])
                speed = self.rng.uniform(U_min, os_csog_state[2] - 2.0)
                heading_modifier = self.rng.uniform(self._config.ot_heading_range[0], self._config.ot_heading_range[1])

            elif scenario_type == ScenarioType.OT_en:
                bearing = self.rng.uniform(self._config.ot_bearing_range[0], self._config.ot_bearing_range[1])
                speed = self.rng.uniform(os_csog_state[2], U_max)
                heading_modifier = self.rng.uniform(self._config.ot_heading_range[0], self._config.ot_heading_range[1])

            elif scenario_type == ScenarioType.CR_GW:
                bearing = self.rng.uniform(self._config.cr_bearing_range[0], self._config.cr_bearing_range[1])
                speed = self.rng.uniform(U_min, U_max)
                heading_modifier = -90.0 + self.rng.uniform(self._config.cr_heading_range[0], self._config.cr_heading_range[1])

            elif scenario_type == ScenarioType.CR_SO:
                bearing = self.rng.uniform(-self._config.cr_bearing_range[1], -self._config.cr_bearing_range[0])
                speed = self.rng.uniform(U_min, U_max)
                heading_modifier = 90.0 + self.rng.uniform(self._config.cr_heading_range[0], self._config.cr_heading_range[1])

            else:
                bearing = self.rng.uniform(0.0, 2.0 * np.pi)
                speed = self.rng.uniform(U_min, U_max)
                heading_modifier = self.rng.uniform(0.0, 359.999)

            bearing = np.deg2rad(bearing)
            heading = os_csog_state[3] + np.deg2rad(heading_modifier)

            distance_os_ts = self.rng.uniform(self._config.dist_between_ships_range[0], self._config.dist_between_ships_range[1])
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
        draft: float = 5.0,
        heading: Optional[float] = None,
        min_land_clearance: float = 100.0,
    ) -> np.ndarray:
        """Creates a random COG-SOG state which adheres to the ship's draft and maximum speed.

        Args:
            - U_min (float, optional): Minimum speed of the ship. Defaults to 1.0.
            - U_max (float, optional): Maximum speed of the ship. Defaults to 15.0.
            - draft (float, optional): How deep the ship keel is into the water. Defaults to 5.
            - heading (Optional[float]): Heading of the ship in radians. Defaults to None.
            - min_land_clearance (float, optional): Minimum distance to land. Defaults to 100.0.

        Returns:
            - np.ndarray: Array containing the vessel state = [x, y, speed, heading]
        """
        x, y = mapf.generate_random_position_from_draft(self.rng, self.enc, draft, min_land_clearance, self.safe_sea_cdt)
        speed = self.rng.uniform(U_min, U_max)
        if heading is None:
            heading = self.rng.uniform(0.0, 2.0 * np.pi)

        return np.array([x, y, speed, heading])

    @property
    def enc_bbox(self) -> np.ndarray:
        """Returns the bounding box of the considered ENC area.

        Returns:
            - np.ndarray: Array containing the ENC bounding box = [min_x, min_y, max_x, max_y]
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
        - scenario_config (ScenarioConfig): Scenario configuration

    Returns:
        - str: The filename of the saved scenario.
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
        - config (ScenarioConfig): Scenario configuration

    Returns:
        - Tuple[np.ndarray, np.ndarray]: Global map origin and size
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
    """Processes AIS data from file, returns a dict containing AIS VesselData, ship MMSIs and the coordinate frame origin and size.

    Args:
        - config (ScenarioConfig): Configuration object containing all parameters/settings related to the creation of a scenario.

    Returns:
        - dict: Dictionary containing processed AIS data.
    """
    output = {}
    if config.ais_data_file is not None:
        output = file_utils.read_ais_data(config.ais_data_file, config.ship_data_file, config.utm_zone, config.map_origin_enu, config.map_size, config.dt_sim)
    return output
