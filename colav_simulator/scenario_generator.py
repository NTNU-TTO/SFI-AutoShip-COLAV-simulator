"""
    scenario_generator.py

    Summary:
        Contains functionality for loading existing scenario definitions,
        and also a ScenarioGenerator class for generating new scenarios. Functionality
        for saving these new scenarios also exists.

    Author: Trym Tengesdal, Joachim Miller, Melih Akdag
"""

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import colav_simulator.behavior_generator as bg
import colav_simulator.common.config_parsing as cp
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp
import colav_simulator.core.ship as ship
import colav_simulator.core.stochasticity as stoch
import colav_simulator.scenario_config as sc
import numpy as np
import seacharts.enc as senc
import shapely.geometry as geometry

np.set_printoptions(suppress=True, formatter={"float_kind": "{:.2f}".format})


@dataclass
class Config:
    """Configuration class for managing all parameters/settings related to the creation of scenarios.
    All angle ranges are in degrees, and all distances are in meters.
    """

    verbose: bool = False
    manual_episode_accept: bool = False  # Whether or not the user has to accept each generated episode of a scenario.
    behavior_generator: bg.Config = field(default_factory=lambda: bg.Config())

    ho_bearing_range: list = field(
        default_factory=lambda: [-20.0, 20.0]
    )  # Range of [min, max] bearing from the own-ship to the target ship for head-on scenarios
    ho_heading_range: list = field(
        default_factory=lambda: [-15.0, 15.0]
    )  # Range of [min, max] heading variations of the target ship relative to completely reciprocal head-on scenarios
    ot_bearing_range: list = field(
        default_factory=lambda: [-20.0, 20.0]
    )  # Range of [min, max] bearing from the own-ship to the target ship for overtaking scenarios
    ot_heading_range: list = field(
        default_factory=lambda: [-15.0, 15.0]
    )  # Range of [min, max] heading variations of the target ship relative to completely parallel overtaking scenarios
    cr_bearing_range: list = field(
        default_factory=lambda: [15.1, 112.5]
    )  # Range of [min, max] bearing from the own-ship to the target ship for crossing scenarios
    cr_heading_range: list = field(
        default_factory=lambda: [-15.0, 15.0]
    )  # Range of [min, max] heading variations of the target ship relative to completely orthogonal crossing scenarios
    dist_between_ships_range: list = field(
        default_factory=lambda: [200, 10000]
    )  # Range of [min, max] distance variations possible between ships.
    csog_state_perturbation_covariance: np.array = field(default_factory=lambda: np.diag([25.0, 25.0, 0.5, 3.0]))
    t_cpa_threshold: float = 1000.0  # Threshold for the maximum time to CPA for vessel pairs in a scenario
    d_cpa_threshold: float = 200.0  # Threshold for the maximum distance to CPA for vessel pairs in a scenario
    scenario_files: Optional[list] = None  # Default list of scenario files to load from.
    scenario_folder: Optional[str] = None  # Default scenario folder to load from.

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config(
            verbose=config_dict["verbose"],
            manual_episode_accept=config_dict["manual_episode_accept"],
            behavior_generator=bg.Config.from_dict(config_dict["behavior_generator"]),
            ho_bearing_range=config_dict["ho_bearing_range"],
            ho_heading_range=config_dict["ho_heading_range"],
            ot_bearing_range=config_dict["ot_bearing_range"],
            ot_heading_range=config_dict["ot_heading_range"],
            cr_bearing_range=config_dict["cr_bearing_range"],
            cr_heading_range=config_dict["cr_heading_range"],
            dist_between_ships_range=config_dict["dist_between_ships_range"],
            t_cpa_threshold=config_dict["t_cpa_threshold"],
            d_cpa_threshold=config_dict["d_cpa_threshold"],
            csog_state_perturbation_covariance=np.diag(config_dict["csog_state_perturbation_covariance"]),
        )
        config.csog_state_perturbation_covariance[3, 3] = np.deg2rad(config.csog_state_perturbation_covariance[3, 3])

        if "scenario_files" in config_dict:
            config.scenario_files = config_dict["scenario_files"]

        if "scenario_folder" in config_dict:
            config.scenario_folder = config_dict["scenario_folder"]
            config.scenario_files = None

        config.behavior_generator = bg.Config.from_dict(config_dict["behavior_generator"])
        return config

    def to_dict(self):
        output = asdict(self)
        output["behavior_generator"] = self.behavior_generator.to_dict()
        output["csog_state_perturbation_covariance"] = self.csog_state_perturbation_covariance.diagonal().tolist()
        output["csog_state_perturbation_covariance"][3] = float(
            np.rad2deg(output["csog_state_perturbation_covariance"][3])
        )
        return output


class ScenarioGenerator:
    """Class for generating maritime traffic scenarios in a given geographical environment."""

    rng: np.random.Generator
    enc: senc.ENC
    behavior_generator: bg.BehaviorGenerator

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
        self._config: Config = Config()
        if config:
            self._config = config
        elif config_file:
            self._config = cp.extract(Config, config_file, dp.scenario_generator_schema)

        self.safe_sea_cdt: Optional[list] = None
        self.safe_sea_cdt_weights: Optional[list] = None

        if init_enc:
            self.enc = senc.ENC(config_file=enc_config_file, **kwargs)
            self._setup_cdt(show_plots=False)

        self.rng = np.random.default_rng(seed=seed)
        self.behavior_generator = bg.BehaviorGenerator(self._config.behavior_generator)

        self._disturbance_handles: list = []
        self._episode_counter: int = 0
        self._uniform_os_state_update_indices: list = []
        self._os_state_update_indices: list = []
        self._os_plan_update_indices: list = []
        self._do_state_update_indices: list = []
        self._do_plan_update_indices: list = []
        self._disturbance_update_indices: list = []
        self._bad_episode: bool = False

        self._prev_disturbance: Optional[stoch.Disturbance] = None
        self._prev_ship_list: list = []
        self._first_csog_states: list = []
        self._position_generation: sc.PositionGenerationMethod = (
            sc.PositionGenerationMethod.UniformInTheMapThenGaussian
        )  # set by scenario config

    def seed(self, seed: Optional[int] = None) -> None:
        """Seeds the random number generator.

        Args:
            seed (Optional[int]): Integer seed. Defaults to None.
        """
        self.rng = np.random.default_rng(seed=seed)
        self.behavior_generator.seed(seed=seed)

    def _setup_cdt(self, vessel_min_depth: int = 5, show_plots: bool = False) -> None:
        """Sets up the constrained Delaunay triangulation for the ENC map, for a vessel minimum depth.

        Args:
            show_plots (bool, optional): Wether to show cdt plots or not. Defaults to False.
        """
        self.safe_sea_cdt = mapf.create_safe_sea_triangulation(
            self.enc, vessel_min_depth=vessel_min_depth, show_plots=show_plots
        )
        self.safe_sea_cdt_weights = mhm.compute_triangulation_weights(self.safe_sea_cdt)

    def _configure_enc(self, scenario_config: sc.ScenarioConfig) -> senc.ENC:
        """Configures the ENC object based on the scenario config file.

        Args:
            - scenario_config (sc.ScenarioConfig): Scenario config object.

        Returns:
            - (senc.ENC): Configured ENC object.
        """
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

    def reset_episode_counter(self, reset: bool) -> None:
        """Resets the episode counter."""
        if reset:
            self._episode_counter = 0

    def determine_indices_of_episode_parameter_updates(self, config: sc.ScenarioConfig) -> None:
        """Determines the episode indices when the OS plan+state, DO state, DO plan and disturbance should be updated/re-randomized.

        Args:
            config (sc.ScenarioConfig): Scenario config object.

        """
        n_episodes = config.episode_generation.n_episodes
        n_constant_os_state_episodes = config.episode_generation.n_constant_os_state_episodes
        n_constant_os_plan_episodes = config.episode_generation.n_constant_os_plan_episodes
        n_constant_do_state_episodes = config.episode_generation.n_constant_do_state_episodes
        n_plans_per_do_state = config.episode_generation.n_plans_per_do_state
        n_constant_disturbance_episodes = config.episode_generation.n_constant_disturbance_episodes
        delta_uniform_position_sample = config.episode_generation.delta_uniform_position_sample
        self._os_state_update_indices = [-1 for _ in range(n_episodes)]
        self._os_plan_update_indices = [-1 for _ in range(n_episodes)]
        self._do_state_update_indices = [-1 for _ in range(n_episodes)]
        self._do_plan_update_indices = [-1 for _ in range(n_episodes)]
        self._disturbance_update_indices = [-1 for _ in range(n_episodes)]
        self._uniform_os_state_update_indices = [-1 for _ in range(n_episodes)]
        for ep in range(n_episodes):
            if ep % delta_uniform_position_sample == 0:
                self._uniform_os_state_update_indices[ep] = ep

            if ep % n_constant_os_state_episodes == 0:
                self._os_state_update_indices[ep] = ep

            if ep % n_constant_os_plan_episodes == 0:
                self._os_plan_update_indices[ep] = ep

            if ep % n_constant_disturbance_episodes == 0:
                self._disturbance_update_indices[ep] = ep

            if ep % (n_plans_per_do_state * n_constant_do_state_episodes) == 0:
                self._do_state_update_indices[ep] = ep

            if ep % n_plans_per_do_state == 0:
                self._do_plan_update_indices[ep] = ep

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

    def load_scenario_from_folder(self, folder: Path, scenario_name: str, show: bool = False) -> Tuple[list, senc.ENC]:
        """Loads all episode files for a given scenario from a folder that match the specified `scenario_name`.

        Args:
            - folder (Path): Path to folder containing scenario files.
            - scenario_name (str): Name of the scenario.
            - show (bool, optional): Flag determining whether or not to show the episode setups through seacharts. Defaults to False.

        Returns:
            - Tuple[list, senc.ENC]: List of scenario files and the corresponding ENC object.
        """
        scenario_episode_list = []
        first = True
        file_list = [file for file in folder.iterdir()]
        file_list.sort(key=lambda x: x.name.split("_")[-3])
        for _, file in enumerate(file_list):
            if not (scenario_name in file.name and file.suffix == ".yaml"):
                continue

            if self._config.verbose:
                print(f"ScenarioGenerator: Loading scenario file: {file.name}...")
            ship_list, disturbance, config = self.load_episode(config_file=file)
            if first:
                first = False
                config.new_load_of_map_data = True
                enc = self._configure_enc(config)
            else:
                config.new_load_of_map_data = False

            scenario_episode_list.append({"ship_list": ship_list, "disturbance": disturbance, "config": config})

            if show:
                self.visualize_episode(ship_list, disturbance, enc, config)

            self._episode_counter += 1

        if self._config.verbose:
            print(f"ScenarioGenerator: Finished loading scenario episode files for scenario: {scenario_name}.")
        if show:
            input("Press enter to continue...")
            enc.close_display()
        return (scenario_episode_list, enc)

    def visualize_disturbance(self, disturbance: stoch.Disturbance | None, enc: senc.ENC) -> None:
        """Visualizes the disturbance object.

        Args:
            disturbance (stoch.Disturbance | None): Disturbance object.
            enc (senc.ENC): _description_
        """
        if disturbance is None:
            return

        ddata = disturbance.get()
        if self._disturbance_handles:
            for handle in self._disturbance_handles:
                handle.remove()

        handles = []
        if ddata.currents is not None and ddata.currents["speed"] > 0.0:
            handles.extend(
                mapf.plot_disturbance(
                    magnitude=70.0,
                    direction=ddata.currents["direction"],
                    name="current: " + str(ddata.currents["speed"]) + " m/s",
                    enc=enc,
                    color="white",
                    linewidth=1.0,
                    location="topright",
                    text_location_offset=(0.0, 0.0),
                )
            )

        if ddata.wind is not None and ddata.wind["speed"] > 0.0:
            handles.extend(
                mapf.plot_disturbance(
                    magnitude=70.0,
                    direction=ddata.wind["direction"],
                    name="wind: " + str(ddata.wind["speed"]) + " m/s",
                    enc=enc,
                    color="peru",
                    linewidth=1.0,
                    location="topright",
                    text_location_offset=(0.0, -20.0),
                )
            )
        self._disturbance_handles = handles

    def visualize_episode(
        self, ship_list: list, disturbance: stoch.Disturbance | None, enc: senc.ENC, config: sc.ScenarioConfig
    ) -> None:
        """Visualizes a fully defined scenario episode.

        Args:
            ship_list (list): List of ships in the scenario with initialized poses and plans.
            disturbance (stoch.Disturbance | None): Disturbance object for the episode.
            enc (senc.ENC): ENC object.
            config (sc.ScenarioConfig): Scenario config object.
        """
        enc.start_display()
        for ship_obj in reversed(ship_list):
            ship_color = "orangered" if ship_obj.id == 0 else "goldenrod"
            plan_color = "orange" if ship_obj.id == 0 else "yellow"
            if ship_obj.waypoints.size > 0:
                mapf.plot_waypoints(
                    ship_obj.waypoints,
                    enc,
                    color=plan_color,
                    point_buffer=2.0,
                    disk_buffer=6.0,
                    hole_buffer=2.0,
                )
            if ship_obj.trajectory.size > 0:
                mapf.plot_trajectory(ship_obj.trajectory, enc, color=plan_color, linewidth=1.0)

            ship_poly = mapf.create_ship_polygon(
                ship_obj.csog_state[0],
                ship_obj.csog_state[1],
                mf.wrap_angle_to_pmpi(ship_obj.csog_state[3]),
                ship_obj.length,
                ship_obj.width,
                5.0,
                5.0,
            )
            enc.draw_polygon(ship_poly, color=ship_color, fill=True, alpha=0.6)

        self.visualize_disturbance(disturbance, enc)

    def load_episode(self, config_file: Path) -> Tuple[list, stoch.Disturbance, sc.ScenarioConfig]:
        """Loads a fully defined scenario episode from configuration file.

        NOTE: The file must have a ship list with fully specified ship configurations,
        and a corresponding correct number of random ships (excluded the own-ship with ID 0).

        NOTE: The scenario ENC object is not initialized here, but in the `load_scenario_from_folder` function.

        Args:
            - config_file (Path): Absolute path to the scenario config file.

        Returns:
            - Tuple[list, sc.ScenarioConfig]: List of ships in the scenario with initialized poses and plans, the final scenario config object.
        """
        config = cp.extract(sc.ScenarioConfig, config_file, dp.scenario_schema)
        ship_list = []
        disturbance = stoch.Disturbance(config.stochasticity) if config.stochasticity is not None else None
        for ship_cfg in config.ship_list:
            assert (
                ship_cfg.csog_state.size > 0
                and (
                    (ship_cfg.waypoints.size > 0 and ship_cfg.speed_plan.size > 0) or ship_cfg.goal_csog_state.size > 0
                )
                and ship_cfg.id >= 0
            ), "A fully specified ship config has an id, initial csog_state, waypoints + speed_plan or goal state."
            ship_obj = ship.Ship(mmsi=ship_cfg.mmsi, identifier=ship_cfg.id, config=ship_cfg)
            ship_list.append(ship_obj)

        return ship_list, disturbance, config

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
        self,
        config: Optional[sc.ScenarioConfig] = None,
        config_file: Optional[Path] = None,
        enc: Optional[senc.ENC] = None,
        new_load_of_map_data: Optional[bool] = None,
        show_plots: Optional[bool] = False,
        save_scenario: Optional[bool] = False,
        save_scenario_folder: Optional[Path] = dp.scenarios,
        reset_episode_counter: Optional[bool] = True,
    ) -> Tuple[list, senc.ENC]:
        """Main class function. Creates a maritime scenario, with a number of `n_episodes` based on the input config or config file.

        If specified, the ENC object provides the geographical environment.

        Args:
            - config (sc.ScenarioConfig, optional): Scenario config object. Defaults to None.
            - config_file (Path, optional): Absolute path to the scenario config file. Defaults to None.
            - enc (ENC, optional): Electronic Navigational Chart object containing the geographical environment. Defaults to None.
            - new_load_of_map_data (bool, optional): Flag determining whether or not to read ENC data from shapefiles again. Defaults to True.
            - show_plots (bool, optional): Flag determining whether or not to show seacharts debugging plots. Defaults to False.
            - save_scenario (bool, optional): Flag determining whether or not to save the scenario definition. Defaults to False.
            - save_scenario_folder (Path, optional): Absolute path to the folder where the scenario definition should be saved. Defaults to dp.scenarios.
            - reset_episode_counter (bool, optional): Flag determining whether or not to reset the episode counter. Defaults to True.

        Returns:
            - Tuple[list, ENC]: List of scenario episodes, each containing a dictionary of episode information. Also, the corresponding ENC object is returned.
        """
        if config is None and config_file is not None:
            config = cp.extract(sc.ScenarioConfig, config_file, dp.scenario_schema)
            config.filename = config_file.name

        if config is None and config_file is None:
            config = cp.extract(sc.ScenarioConfig, self.create_file_path_list_from_config()[0], dp.scenario_schema)

        assert config is not None, "Config should not be none here."
        self.reset_episode_counter(reset_episode_counter)
        show_plots = True if self._config.manual_episode_accept else show_plots
        save_scenario = save_scenario if save_scenario is not None else config.save_scenario
        ais_vessel_data_list = []
        mmsi_list = []
        ais_data_output = sc.process_ais_data(config)
        if ais_data_output:
            ais_vessel_data_list = ais_data_output["vessels"]
            mmsi_list = ais_data_output["mmsi_list"]
            config.map_origin_enu = ais_data_output["map_origin_enu"]
            config.map_size = ais_data_output["map_size"]
        config.map_origin_enu, config.map_size = sc.find_global_map_origin_and_size(config)
        if new_load_of_map_data is not None:
            config.new_load_of_map_data = new_load_of_map_data

        if enc is not None:
            self.enc = enc
            enc_copy = copy.deepcopy(enc)
        else:
            enc_copy = self._configure_enc(config)
        self._setup_cdt(show_plots=False)

        if config.n_random_ships is not None:
            n_random_ships = config.n_random_ships
        elif config.n_random_ships_range is not None:
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

        n_episodes = config.episode_generation.n_episodes
        self.determine_indices_of_episode_parameter_updates(config)
        self._position_generation = config.episode_generation.position_generation
        self.behavior_generator.reset()
        scenario_episode_list = []
        for ep in range(n_episodes):
            if show_plots:
                self.enc.start_display()

            episode = {}
            episode["ship_list"], episode["disturbance"], episode["config"] = self.generate_episode(
                copy.deepcopy(ship_list),
                copy.deepcopy(config),
                ais_vessel_data_list,
                mmsi_list,
                show_plots=show_plots,
            )
            if self._bad_episode:
                continue

            ep_str = str(ep + 1).zfill(3)
            episode["config"].name = f"{config.name}_ep{ep_str}"
            if self._config.manual_episode_accept:
                print("ScenarioGenerator: Accept episode? (y/n)")
                answer = input()  # "y"
                if answer not in ["y", "Y", "yes", "Yes"]:
                    if ep < n_episodes - 1:
                        self._uniform_os_state_update_indices[ep + 1] = ep + 1
                        self._os_plan_update_indices[ep + 1] = ep + 1
                        self._os_state_update_indices[ep + 1] = ep + 1
                        self._do_plan_update_indices[ep + 1] = ep + 1
                    continue

            if self._config.verbose:
                print(f"ScenarioGenerator: Episode {ep + 1} of {n_episodes} created.")

            if save_scenario:
                episode["config"].filename = sc.save_scenario_episode_definition(
                    episode["config"], save_scenario_folder
                )

            self._episode_counter += 1
            scenario_episode_list.append(episode)

        if show_plots:
            input(
                "Press enter to continue. Will take a while to load plots if you generated 500+ episodes with visualization on..."
            )
            self.enc.close_display()
        return scenario_episode_list, enc_copy

    def generate_episode(
        self,
        ship_list: list,
        config: sc.ScenarioConfig,
        ais_vessel_data_list: Optional[list],
        mmsi_list: Optional[list],
        show_plots: Optional[bool] = True,
    ) -> Tuple[list, Optional[stoch.Disturbance], sc.ScenarioConfig]:
        """Creates a single maritime scenario episode.

                Some ships in the episode can be partially or fully specified by the AIS ship data, if not none.

                Random plans for each ship will be created unless specified in ship_list entries or loaded from AIS data.
        >
                Args:
                    - ship_list (list): List of ships to be considered in simulation.
                    - config (sc.ScenarioConfig): Scenario config object.
                    - ais_vessel_data_list (Optional[list]): Optional list of AIS vessel data objects.
                    - mmsi_list (Optional[list]): Optional list of corresponding MMSI numbers for the AIS vessels.
                    - show_plots (Optional[bool]): Flag determining whether or not to show seacharts debugging plots. Defaults to False.

                Returns:
                    - Tuple[list, Optional[stoch.Disturbance], sc.ScenarioConfig]: List of ships in the scenario with initialized poses and plans, the disturbance object for the episode (if specified) and the final scenario config object.
        """
        ship_replan_flags = self.determine_replanning_flags(ship_list, config)

        ship_list, config = self.transfer_vessel_ais_data(ship_list, config, ais_vessel_data_list, mmsi_list)

        ship_list, config, _ = self.generate_ship_csog_states(ship_list, config)

        self.behavior_generator.setup(
            self.rng,
            ship_list,
            ship_replan_flags,
            self.enc,
            self.safe_sea_cdt,
            self.safe_sea_cdt_weights,
            config.t_end - config.t_start,
            show_plots=show_plots,
        )
        ship_list, config.ship_list = self.behavior_generator.generate(
            self.rng,
            ship_list,
            config.ship_list,
            simulation_timespan=config.t_end - config.t_start,
            show_plots=show_plots,
        )

        ship_list.sort(key=lambda x: x.id)
        config.ship_list.sort(key=lambda x: x.id)

        disturbance = self.generate_disturbance(config)

        self._bad_episode = self.check_for_bad_episode(ship_list)

        self._prev_ship_list = copy.deepcopy(ship_list)
        return ship_list, disturbance, config

    def check_for_bad_episode(
        self, ship_list: list, minimum_plan_length: float = 200.0, next_wp_angle_threshold: float = np.deg2rad(120.0)
    ) -> bool:
        """Checks if the episode is bad, i.e. if any of the ships are outside the map,
        the plan is less than the minimum length.

        Args:
            ship_list (list): List of ships to be considered in simulation.
            minimum_plan_length (float, optional): Minimum length of the plan. Defaults to 200.0.
            next_wp_angle_threshold (float, optional): Threshold for the angle between the LOS to the next waypoint and the ship heading. Defaults to 120.0.

        Returns:
            bool: True if the episode is bad, False otherwise.
        """
        for ship_obj in ship_list:
            if not mapf.point_in_polygon_list(
                geometry.Point(ship_obj.csog_state[1], ship_obj.csog_state[0]), self.safe_sea_cdt
            ):
                return True

            path_length = np.sum(np.linalg.norm(np.diff(ship_obj.waypoints, axis=1), axis=0))
            if path_length < minimum_plan_length:
                return True

            # next_wp = ship_obj.waypoints[:, 1]
            # los_next_wp = np.arctan2(next_wp[1] - ship_obj.csog_state[1], next_wp[0] - ship_obj.csog_state[0])
            # if np.abs(mf.wrap_angle_diff_to_pmpi(los_next_wp, ship_obj.csog_state[3])) > next_wp_angle_threshold:
            #     return True
        return False

    def determine_replanning_flags(self, ship_list: list, config: sc.ScenarioConfig) -> list:
        """Determines the flags for whether or not to generate a new plan for each ship.

        Args:
            ship_list (list): List of ships to be considered in simulation.
            config (sc.ScenarioConfig): Scenario config object.

        Returns:
            list: List of booleans determining whether or not to generate a new plan for each ship.
        """
        replan_list = [False for _ in range(len(ship_list))]
        ep = self._episode_counter
        uniform_in_map_sample = ep % config.episode_generation.delta_uniform_position_sample == 0
        for ship_cfg_idx, _ in enumerate(config.ship_list):
            if uniform_in_map_sample:
                replan_list[ship_cfg_idx] = True

            elif ship_cfg_idx == 0 and ep == self._os_plan_update_indices[ep]:
                replan_list[ship_cfg_idx] = True

            elif ship_cfg_idx > 0 and ep == self._do_plan_update_indices[ep]:
                replan_list[ship_cfg_idx] = True
        return replan_list

    def transfer_vessel_ais_data(
        self,
        ship_list: list,
        config: sc.ScenarioConfig,
        ais_vessel_data_list: Optional[list],
        mmsi_list: Optional[list],
    ) -> Tuple[list, sc.ScenarioConfig]:
        """Transfers AIS vessel data to the ship objects and ship configurations, if available.

        Args:
            - ship_list (list): List of ships to be considered in simulation.
            - config (sc.ScenarioConfig): Scenario config object.
            - ais_vessel_data_list (Optional[list]): Optional list of AIS vessel data objects.
            - mmsi_list (Optional[list]): Optional list of corresponding MMSI numbers for the AIS vessels.

        Returns:
            - Tuple[list, sc.ScenarioConfig]: List of partially initialized ships in the scenario, and the corresponding updated scenario config object.
        """
        if not (ais_vessel_data_list or mmsi_list):
            return ship_list, config

        for ship_cfg_idx, ship_config in enumerate(config.ship_list):
            use_ais_ship_trajectory = True
            if ship_config.random_generated:
                continue

            # The own-ship (with index 0) will not use the predefined AIS trajectory, but can use the AIS data
            # for the initial state.
            idx = 0
            if ship_cfg_idx == 0:
                use_ais_ship_trajectory = False

            if ship_config.mmsi in mmsi_list:
                idx = [i for i in range(len(ais_vessel_data_list)) if ais_vessel_data_list[i].mmsi == ship_config.mmsi][
                    0
                ]

            ais_vessel = ais_vessel_data_list.pop(idx)
            while ais_vessel.status.value not in config.allowed_nav_statuses:
                ais_vessel = ais_vessel_data_list.pop(idx)

            ship_list[ship_cfg_idx].transfer_vessel_ais_data(
                ais_vessel, use_ais_ship_trajectory, ship_config.t_start, ship_config.t_end
            )
            ship_config.csog_state = ship_list[ship_cfg_idx].csog_state
            ship_config.mmsi = ship_list[ship_cfg_idx].mmsi

        return ship_list, config

    def generate_disturbance(self, config: sc.ScenarioConfig) -> Optional[stoch.Disturbance]:
        """Generates a disturbance object from the scenario config.

        Args:
            - config (sc.ScenarioConfig): Scenario config object.

        Returns:
            - stoch.Disturbance: Disturbance object.
        """
        if config.stochasticity is None:
            return None

        ep = self._episode_counter
        if ep == self._disturbance_update_indices[ep]:
            disturbance = stoch.Disturbance(config.stochasticity)
        else:
            disturbance = self._prev_disturbance
        self._prev_disturbance = copy.deepcopy(disturbance)
        return disturbance

    def generate_ship_csog_states(
        self, ship_list: list, config: sc.ScenarioConfig
    ) -> Tuple[list, sc.ScenarioConfig, list]:
        """Generates the initial ship poses for the scenario episode.

        Args:
            ship_list (list): List of ships to be considered in simulation.
            config (sc.ScenarioConfig): Scenario config object.

        Returns:
            Tuple[list, sc.ScenarioConfig, list]: List of partially initialized ships in the scenario with poses set, the updated scenario config object and list of generated/set csog states.
        """
        csog_state_list = []
        ep = self._episode_counter
        uniform_in_map_sample = self._uniform_os_state_update_indices[ep] == ep
        for ship_cfg_idx, ship_config in enumerate(config.ship_list):
            if ship_config.csog_state is not None:
                csog_state_list.append(ship_config.csog_state)
                continue

            ship_obj = ship_list[ship_cfg_idx]
            # Use 90% of the maximum speed as the maximum speed for the ships

            if ship_cfg_idx == 0:
                if ep == self._os_state_update_indices[ep]:
                    csog_state = self.generate_random_csog_state(
                        U_min=2.0,
                        U_max=0.9 * ship_obj.max_speed,
                        draft=ship_obj.draft,
                        min_land_clearance=np.min([30.0, ship_obj.length * 3.0]),
                        first_episode_csog_state=self._first_csog_states[ship_cfg_idx]
                        if not uniform_in_map_sample
                        else None,
                    )
                else:
                    csog_state = self._prev_ship_list[ship_cfg_idx].csog_state
            new_uniform_os_state = ep == self._os_state_update_indices[ep] and uniform_in_map_sample
            if ship_cfg_idx > 0:
                if ep == self._do_state_update_indices[ep] or new_uniform_os_state:
                    csog_state = self.generate_target_ship_csog_state(
                        config.type,
                        csog_state_list[0],
                        U_min=2.0,
                        U_max=0.9 * ship_obj.max_speed,
                        draft=ship_obj.draft,
                        min_land_clearance=np.min([30.0, ship_obj.length * 3.0]),
                        t_cpa_threshold=self._config.t_cpa_threshold,
                        d_cpa_threshold=self._config.d_cpa_threshold,
                        first_episode_csog_state=self._first_csog_states[ship_cfg_idx]
                        if not uniform_in_map_sample
                        else None,
                    )
                else:
                    csog_state = self._prev_ship_list[ship_cfg_idx].csog_state

            ship_config.csog_state = csog_state
            ship_obj.set_initial_state(ship_config.csog_state)
            csog_state_list.append(ship_config.csog_state)

        if ep % config.episode_generation.delta_uniform_position_sample == 0:
            self._first_csog_states = csog_state_list
        return ship_list, config, csog_state_list

    def generate_target_ship_csog_state(
        self,
        scenario_type: sc.ScenarioType,
        os_csog_state: np.ndarray,
        U_min: float = 1.0,
        U_max: float = 10.0,
        draft: float = 2.0,
        min_land_clearance: float = 50.0,
        t_cpa_threshold: float = 1000.0,
        d_cpa_threshold: float = 100.0,
        first_episode_csog_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generates a position for the target ship based on the perspective of the first ship/own-ship,
        such that the scenario is of the input type.

        Args:
            - scenario_type (sc.ScenarioType): Type of scenario.
            - os_csog_state (np.ndarray): Own-ship COG-SOG state = [x, y, speed, heading].
            - U_min (float, optional): Obstacle minimum speed. Defaults to 1.0.
            - U_max (float, optional): Obstacle maximum speed. Defaults to 10.0.
            - draft (float, optional): Draft of target ship. Defaults to 2.0.
            - min_land_clearance (float, optional): Minimum distance between target ship and land. Defaults to 100.0.
            - t_cpa_threshold (float, optional): Time to CPA threshold. Defaults to 1000.0.
            - d_cpa_threshold (float, optional): Distance to CPA threshold. Defaults to 100.0.
            - first_episode_csog_state (Optional[np.ndarray], optional): First scenario episode target ship COG-SOG state. Defaults to None.

        Returns:
            - np.ndarray: Target ship position = [x, y].
        """
        if (
            first_episode_csog_state is not None
            and self._position_generation == sc.PositionGenerationMethod.UniformInTheMapThenGaussian
        ):
            return self.generate_gaussian_csog_state(
                first_episode_csog_state, self._config.csog_state_perturbation_covariance, draft
            )

        if any(np.isnan(os_csog_state)):
            return self.generate_random_csog_state(
                U_min=U_min, U_max=U_max, draft=draft, min_land_clearance=min_land_clearance
            )

        if scenario_type == sc.ScenarioType.MS:
            scenario_type = self.rng.choice(
                [
                    sc.ScenarioType.HO,
                    sc.ScenarioType.OT_ing,
                    sc.ScenarioType.OT_en,
                    sc.ScenarioType.CR_GW,
                    sc.ScenarioType.CR_SO,
                ]
            )

        if scenario_type == sc.ScenarioType.OT_en and U_max - 2.0 <= os_csog_state[2]:
            print(
                "WARNING: sc.ScenarioType = OT_en: Own-ship speed should be below the maximum target ship speed minus margin of 2.0. Selecting a different scenario type..."
            )
            scenario_type = self.rng.choice(
                [sc.ScenarioType.HO, sc.ScenarioType.OT_ing, sc.ScenarioType.CR_GW, sc.ScenarioType.CR_SO]
            )

        if scenario_type == sc.ScenarioType.OT_ing and U_min >= os_csog_state[2] - 2.0:
            print(
                "WARNING: sc.ScenarioType = OT_ing: Own-ship speed minus margin of 2.0 should be above the minimum target ship speed. Selecting a different scenario type..."
            )
            scenario_type = self.rng.choice(
                [sc.ScenarioType.HO, sc.ScenarioType.OT_en, sc.ScenarioType.CR_GW, sc.ScenarioType.CR_SO]
            )

        depth = mapf.find_minimum_depth(draft, self.enc)
        safe_sea = self.enc.seabed[depth]
        max_iter = 5000
        y_min, x_min, y_max, x_max = self.enc.bbox
        distance_os_ts = self.rng.uniform(
            self._config.dist_between_ships_range[0], self._config.dist_between_ships_range[1]
        )
        x = os_csog_state[0] + distance_os_ts * np.cos(os_csog_state[3] + np.pi / 2.0)
        y = os_csog_state[1] + distance_os_ts * np.sin(os_csog_state[3] + np.pi / 2.0)
        speed = self.rng.uniform(U_min, U_max)
        accepted = False
        for i in range(max_iter):
            if scenario_type == sc.ScenarioType.HO:
                bearing = self.rng.uniform(self._config.ho_bearing_range[0], self._config.ho_bearing_range[1])
                speed = self.rng.uniform(U_min, U_max)
                heading_modifier = 180.0 + self.rng.uniform(
                    self._config.ho_heading_range[0], self._config.ho_heading_range[1]
                )

            elif scenario_type == sc.ScenarioType.OT_ing:
                bearing = self.rng.uniform(self._config.ot_bearing_range[0], self._config.ot_bearing_range[1])
                speed = self.rng.uniform(U_min, os_csog_state[2] - 2.0)
                heading_modifier = self.rng.uniform(self._config.ot_heading_range[0], self._config.ot_heading_range[1])

            elif scenario_type == sc.ScenarioType.OT_en:
                bearing = self.rng.uniform(self._config.ot_bearing_range[0], self._config.ot_bearing_range[1])
                speed = self.rng.uniform(os_csog_state[2], U_max)
                heading_modifier = self.rng.uniform(self._config.ot_heading_range[0], self._config.ot_heading_range[1])

            elif scenario_type == sc.ScenarioType.CR_GW:
                bearing = self.rng.uniform(self._config.cr_bearing_range[0], self._config.cr_bearing_range[1])
                speed = self.rng.uniform(U_min, U_max)
                heading_modifier = -90.0 + self.rng.uniform(
                    self._config.cr_heading_range[0], self._config.cr_heading_range[1]
                )

            elif scenario_type == sc.ScenarioType.CR_SO:
                bearing = self.rng.uniform(-self._config.cr_bearing_range[1], -self._config.cr_bearing_range[0])
                speed = self.rng.uniform(U_min, U_max)
                heading_modifier = 90.0 + self.rng.uniform(
                    self._config.cr_heading_range[0], self._config.cr_heading_range[1]
                )

            else:
                bearing = self.rng.uniform(0.0, 2.0 * np.pi)
                speed = self.rng.uniform(U_min, U_max)
                heading_modifier = self.rng.uniform(0.0, 359.999)

            bearing = np.deg2rad(bearing)
            heading = os_csog_state[3] + np.deg2rad(heading_modifier)

            distance_os_ts = self.rng.uniform(
                self._config.dist_between_ships_range[0], self._config.dist_between_ships_range[1]
            )
            x = os_csog_state[0] + distance_os_ts * np.cos(os_csog_state[3] + bearing)
            y = os_csog_state[1] + distance_os_ts * np.sin(os_csog_state[3] + bearing)

            inside_bbox = mhm.inside_bbox(np.array([x, y]), (x_min, y_min, x_max, y_max))
            risky_enough = mhm.check_if_situation_is_risky_enough(
                os_csog_state, np.array([x, y, speed, heading]), t_cpa_threshold, d_cpa_threshold
            )
            # hazard_between_ships = mapf.check_if_segment_crosses_grounding_hazards(
            #     self.enc, np.array([x, y]), os_csog_state[:2]
            # )

            if (
                risky_enough and safe_sea.geometry.contains(geometry.Point(y, x)) and inside_bbox
            ):  # and not hazard_between_ships:
                accepted = True
                break
        if not accepted:
            print(
                "WARNING: Could not find an acceptable starting state for the target ship. Using a random state projected onto the safe sea.."
            )
            # self.enc.draw_circle((y, x), radius=10.0, color="orange", fill=True, alpha=0.6)
            start_pos = np.array([x, y]) + speed * 500.0 * np.array([np.cos(heading), np.sin(heading)])
            end_pos = np.array([x, y])
            new_start_pos = mapf.find_closest_collision_free_point_on_segment(
                self.enc, start_pos, end_pos, draft, min_dist=min_land_clearance
            )
            x, y = new_start_pos[0], new_start_pos[1]
            # self.enc.draw_circle((y, x), radius=10.0, color="red", fill=True, alpha=0.6)
        return np.array([x, y, speed, heading])

    def generate_gaussian_csog_state(self, mean: np.ndarray, cov: np.ndarray, draft: float) -> np.ndarray:
        """Generates a COG-SOG state from a Gaussian distribution around the input mean (first episodic csog state) and covariance.

        Args:
            mean (np.ndarray): Mean of the Gaussian distribution, i.e. the first episodic csog state = [x, y, speed, heading].
            cov (np.ndarray): Covariance of the Gaussian distribution.
            draft (float, optional): Draft of ship. Defaults to 2.0.

        Returns:
            np.ndarray: Array containing the random vessel state = [x, y, speed, heading]
        """
        perturbed_state = self.rng.multivariate_normal(mean, cov)
        safe_sea = self.enc.seabed[mapf.find_minimum_depth(draft, self.enc)]
        max_iter = 2000
        for _ in range(max_iter):
            if safe_sea.geometry.contains(geometry.Point(perturbed_state[1], perturbed_state[0])):
                break
            perturbed_state = self.rng.multivariate_normal(mean, cov)
        return perturbed_state

    def generate_random_csog_state(
        self,
        U_min: float = 1.0,
        U_max: float = 10.0,
        draft: float = 5.0,
        heading: Optional[float] = None,
        min_land_clearance: float = 50.0,
        first_episode_csog_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Creates a random COG-SOG state which adheres to the ship's draft and maximum speed.

        Args:
            - U_min (float, optional): Minimum speed of the ship. Defaults to 1.0.
            - U_max (float, optional): Maximum speed of the ship. Defaults to 10.0.
            - draft (float, optional): How deep the ship keel is into the water. Defaults to 5.
            - heading (Optional[float]): Heading of the ship in radians. Defaults to None.
            - min_land_clearance (float, optional): Minimum distance between ship and land. Defaults to 50.0.
            - first_episode_csog_state (Optional[np.ndarray], optional): First scenario episode ship COG-SOG state. Defaults to None.

        Returns:
            - np.ndarray: Array containing the vessel state = [x, y, speed, heading]
        """
        if (
            first_episode_csog_state is not None
            and self._position_generation == sc.PositionGenerationMethod.UniformInTheMapThenGaussian
        ):
            return self.generate_gaussian_csog_state(
                first_episode_csog_state, self._config.csog_state_perturbation_covariance, draft
            )

        x, y = mapf.generate_random_position_from_draft(
            self.rng, self.enc, draft, self.safe_sea_cdt, self.safe_sea_cdt_weights, min_land_clearance
        )
        speed = self.rng.uniform(U_min, U_max)
        distance_vectors = mapf.compute_distance_vectors_to_grounding(
            np.array([y, x]).reshape(-1, 1), mapf.find_minimum_depth(draft, self.enc), self.enc
        )
        dist_vec = distance_vectors[:, 0]
        angle_to_land = np.arctan2(dist_vec[0], dist_vec[1])
        dist_vec_to_bbox = mapf.compute_distance_vector_to_bbox(y, x, self.enc.bbox, self.enc)
        angle_to_bbox = np.arctan2(dist_vec_to_bbox[0], dist_vec_to_bbox[1])
        if heading is None:
            heading = self.rng.uniform(0.0, 2.0 * np.pi)
            if np.linalg.norm(dist_vec) < 2.0 * min_land_clearance:
                heading = angle_to_land + np.pi + self.rng.uniform(-np.pi / 2.0, np.pi / 2.0)

            if np.linalg.norm(dist_vec_to_bbox) < 2.0 * min_land_clearance:
                heading = angle_to_bbox + np.pi + self.rng.uniform(-np.pi / 2.0, np.pi / 2.0)

        return np.array([x, y, speed, mf.wrap_angle_to_pmpi(heading)])

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
