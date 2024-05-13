"""
    simulator.py

    Summary:
        Contains class definitions for the simulator, enables the
        simulation of a diverse set of COLAV scenarios from their definitions.

    Author: Trym Tengesdal, Magne Aune, Joachim Miller
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp
import colav_simulator.core.stochasticity as stochasticity
import colav_simulator.scenario_config as sc
import colav_simulator.viz.visualizer as viz
import numpy as np
import pandas as pd
import seacharts.enc as senc
import yaml
from colav_simulator.core.ship import Ship

np.set_printoptions(suppress=True, formatter={"float_kind": "{:.4f}".format})


@dataclass
class Config:
    """Simulation related configuration/parameter class."""

    save_scenario_results: bool
    verbose: bool
    visualizer: viz.Config

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config(
            save_scenario_results=config_dict["save_scenario_results"],
            verbose=config_dict["verbose"],
            visualizer=viz.Config.from_dict(config_dict["visualizer"]),
        )
        return config

    @classmethod
    def from_file(cls, config_file: Path):
        assert config_file.exists(), f"Configuration file {config_file} does not exist."
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


class Simulator:
    """Class for simulating collision avoidance/maritime vessel scenarios."""

    def __init__(
        self, config: Optional[Config] = None, config_file: Optional[Path] = dp.simulator_config, **kwargs
    ) -> None:
        """Initializes the simulator.

        Additional key-value arguments can be passed to override the settings in the config file:

        Args:
            config (Optional[Config]): Configuration object. Defaults to None.
            config_file (Optional[Path]): Path to configuration file. Defaults to dp.simulator_config.
            kwargs: key-value dictionary of settings to override. This includes:
                scenario_files (List[str]): List of scenario files to run.

        """
        if config is not None:
            self._config = config
        elif config_file is not None:
            self._config = cp.extract(Config, config_file, dp.simulator_schema, **kwargs)
        else:
            raise ValueError("No configuration file or configuration object provided.")

        self.visualizer: viz.Visualizer = viz.Visualizer(self._config.visualizer)
        self.config: Config = self._config

        self.ownship: Ship = None
        self.ship_list: List[Ship] = []
        self.disturbance: Optional[stochasticity.Disturbance] = None
        self.enc: senc.ENC = None
        self.sconfig: sc.ScenarioConfig = None
        self.relevant_grounding_hazards: list = []
        self.recent_sensor_measurements: list = []

        self.t = 0.0
        self.t_start = 0.0
        self.t_end = 0.0
        self.dt = 0.0
        self.timestamp_start = 0

    def toggle_liveplot_visibility(self, visible: bool) -> None:
        """Toggles the visibility of the live plot.

        Args:
            visible (bool): Whether the live plot should be visible or not.
        """
        self.visualizer.toggle_liveplot_visibility(visible)

    def initialize_scenario_episode(
        self,
        ship_list: List[Ship],
        sconfig: sc.ScenarioConfig,
        enc: senc.ENC,
        disturbance: Optional[stochasticity.Disturbance] = None,
        colav_systems: Optional[list] = None,
        seed: int | None = None,
    ) -> None:
        """Initializes the simulation through setting relevant internal state objects.

        Args:
            - ship_list (List[Ship]): 1 x n_ships array of configured Ship objects. Each ship
            is assumed to be properly configured and initialized to its initial state at
            the scenario start (t0).
            - sconfig (ScenarioConfig): Scenario episode configuration object.
            - enc (senc.ENC): ENC object relevant for the scenario.
            - disturbance (Optional[stochasticity.Disturbance]): Disturbance object relevant for the scenario. Defaults to None.
            - colav_systems (Optional[list]): List of tuples (ship ID, COLAV system) to use for the selected ships involved in the scenario, overrides the existing ones. Defaults to None.
            - seed (int | None): Seed for the random number generator. Defaults to None.
        """
        self.ship_list = ship_list
        self.sconfig = sconfig
        self.enc = enc
        self.disturbance = disturbance
        self.ownship = ship_list[0]
        if colav_systems is not None:
            for ship_id, colav_system in colav_systems:
                for _, ship_obj in enumerate(self.ship_list):
                    if ship_obj.id == ship_id:
                        ship_obj.set_colav_system(colav_system)

        ownship_min_depth = mapf.find_minimum_depth(self.ownship.draft, self.enc)
        self.relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards(ownship_min_depth, self.enc)

        for ship_obj in self.ship_list:
            ship_obj.reset(seed=seed)

        self.timestamp_start = mhm.current_utc_timestamp()
        self.t = sconfig.t_start
        self.t_start = sconfig.t_start
        self.t_end = sconfig.t_end
        self.dt = sconfig.dt_sim
        self.recent_sensor_measurements: list = [None] * len(self.ship_list)
        # print(
        #     f"Initialized scenario ep={sconfig.name} with {len(self.ship_list)} ships and Disturbance={disturbance is not None}."
        # )

    def run(
        self,
        scenario_data_list: list,
        colav_systems: Optional[list] = None,
        terminate_on_collision_or_grounding: bool = True,
    ) -> list:
        """Runs through all specified scenarios with their number of episodes. If none are specified, the scenarios are generated from the config file and run through.

        Seeds for the random number generator are set and incremented for each scenario episode.

        Args:
            - scenario_data_list (list): Premade list of created/configured scenarios. Each entry contains a list of ship objects, scenario configuration objects and relevant ENC objects.
            - colav_systems (Optional[list]): List of tuples (ship ID, COLAV system) to use for the selected ships involved in the scenario, overrides the existing ones. Defaults to None.
            - terminate_on_collision_or_grounding (bool): Whether to terminate the simulation if a collision or grounding occurs.

        Returns:
            list: List of dictionaries containing the following simulation data for each scenario:
            - episode_simdata_list (list): List of dictionaries containing the following simulation data for each scenario episode:
                - vessel_data (list): List of data containers containing the vessel simulation data for each ship (used for evaluation).
                - sim_data (pd.DataFrame): Dataframe containing the simulated data for each ship.
                - ship_info (pd.DataFrame): Dataframe containing the ship info for each ship.
            - enc (senc.Enc): ENC object used in all the scenario episodes.
        """

        if self._config.verbose:
            print("\rSimulator: Started running through scenarios...")

        seed_val = 0

        scenario_simdata_list = []
        for i, (scenario_episode_list, scenario_enc) in enumerate(scenario_data_list):
            scenario_simdata = {}
            if self._config.verbose:
                print(f"\rSimulator: Running scenario nr {i + 1}...")

            episode_simdata_list = []
            for ep, episode_data in enumerate(scenario_episode_list):
                episode_simdata = {}
                ship_list = episode_data["ship_list"]
                episode_disturbance = episode_data["disturbance"]
                episode_config = episode_data["config"]
                scenario_episode_file = episode_config.filename

                self.initialize_scenario_episode(
                    ship_list, episode_config, scenario_enc, episode_disturbance, colav_systems, seed=seed_val
                )

                if self._config.verbose:
                    print(f"\rSimulator: Running scenario episode nr {ep + 1}: {scenario_episode_file}...")
                sim_data, ship_info, sim_times = self.run_scenario_episode(terminate_on_collision_or_grounding)
                if self._config.verbose:
                    print(
                        f"\rSimulator: Finished running through scenario episode nr {ep + 1}: {scenario_episode_file}."
                    )

                self.visualizer.visualize_results(
                    scenario_enc,
                    ship_list,
                    sim_data,
                    sim_times,
                    save_file_path=dp.figure_output / episode_config.name,
                )

                self.visualizer.save_live_plot_animation(dp.animation_output / (episode_config.name + ".gif"))

                vessel_data = mhm.convert_simulation_data_to_vessel_data(sim_data, ship_info, episode_config.utm_zone)

                episode_simdata["vessel_data"] = vessel_data
                episode_simdata["sim_data"] = sim_data
                episode_simdata["ship_info"] = ship_info
                episode_simdata_list.append(episode_simdata)

                seed_val += 1

            if self._config.verbose:
                print(f"\rSimulator: Finished running through episodes of scenario nr {i + 1}.")

            scenario_simdata["episode_simdata_list"] = episode_simdata_list
            scenario_simdata["enc"] = scenario_enc
            scenario_simdata_list.append(scenario_simdata)

        if self._config.verbose:
            print("\rSimulator: Finished running through scenarios.")
        return scenario_simdata_list

    def is_terminated(self, verbose: bool = False, terminate_on_collision_or_grounding: bool = True) -> bool:
        """Check whether the current own-ship state is a terminal state.

        Args:
            verbose (bool): Whether to print out the reason for the termination.

        Returns:
            bool: Whether the current own-ship state is a terminal state
        """
        goal_reached = self.determine_ship_goal_reached(ship_idx=0)
        collided = self.determine_ship_collision(ship_idx=0) and terminate_on_collision_or_grounding
        grounded = self.determine_ship_grounding(ship_idx=0) and terminate_on_collision_or_grounding
        if verbose and collided:
            print(f"Collision at t = {self.t}!")
        if verbose and grounded:
            print(f"Grounding at t = {self.t}!")
        if verbose and goal_reached:
            print(f"Goal reached at t = {self.t}!")
        return collided or grounded or goal_reached

    def is_truncated(self, verbose: bool = False) -> bool:
        """Check whether the current own-ship state is a truncated state (time limit reached).

        Args:
            verbose (bool): Whether to print out the reason for the truncation.

        Returns:
            bool: Whether the current own-ship state is a truncated state
        """
        truncated = self.t > self.t_end
        if verbose and truncated:
            print("Time limit reached!")
        return truncated

    def run_scenario_episode(
        self, terminate_on_collision_or_grounding: bool = True
    ) -> Tuple[pd.DataFrame, dict, np.ndarray]:
        """Runs the simulator for a scenario episode specified by the ship object array, using a time step dt_sim.

        Args:
            terminate_on_collision_or_grounding (bool): Whether to terminate the simulation if a collision or grounding occurs.

        Returns: a tuple containing:
            - sim_data (DataFrame): Dataframe/table containing the ship simulation data.
            - ship_info (dict): Dictionary containing the ship info for each ship.
            - sim_times (np.array): Array containing the simulation times.
        """

        self.visualizer.init_live_plot(self.enc, self.ship_list)

        sim_data = []
        ship_info = {}
        for i, ship_obj in enumerate(self.ship_list):
            ship_info[f"Ship{i}"] = ship_obj.get_ship_info()

        t_end = self.t_end
        while self.t < self.t_end:
            sim_data_dict = self.step()

            sim_data.append(sim_data_dict)

            self.visualizer.update_live_plot(self.t, self.enc, self.ship_list, self.recent_sensor_measurements[0])

            terminated = self.is_terminated(
                verbose=True, terminate_on_collision_or_grounding=terminate_on_collision_or_grounding
            )
            truncated = self.is_truncated(verbose=True)
            if terminated or truncated:
                t_end = self.t
                break

        self.visualizer.close_live_plot()

        sim_times = np.arange(self.t_start, t_end, self.dt)
        return pd.DataFrame(sim_data), ship_info, sim_times

    def step(self, remote_actor: bool = False) -> dict:
        """Step through the simulation by one time step.

        Args:
            remote_actor (bool, optional): Whether the own-ship is controlled by a remote actor, i.e. references are set externally from the Ship object. Defaults to False.

        Returns:
            dict: Dictionary containing the current time step simulation data for each ship and the disturbance data if applicable.
        """
        sim_data_dict = {}

        disturbance_data: Optional[stochasticity.DisturbanceData] = None
        if self.disturbance is not None:
            disturbance_data = self.disturbance.get()
            self.disturbance.update(self.t, self.dt)
            sim_data_dict["currents"] = disturbance_data.currents
            sim_data_dict["wind"] = disturbance_data.wind
            sim_data_dict["waves"] = disturbance_data.waves

        true_do_states = mhm.extract_do_states_from_ship_list(self.t, self.ship_list)
        for i, ship_obj in enumerate(self.ship_list):
            tracks, sensor_measurements_i = [], []
            if i == 0:
                relevant_true_do_states = mhm.get_relevant_do_states(true_do_states, i)
                tracks, sensor_measurements_i = ship_obj.track_obstacles(self.t, self.dt, relevant_true_do_states)

            self.recent_sensor_measurements[i] = extract_valid_sensor_measurements(
                self.t, self.recent_sensor_measurements[i], sensor_measurements_i
            )

            # Plans a decision for the ship depending on its configuration
            if ship_obj.t_start <= self.t:
                if not (i == 0 and remote_actor):  # Skip own-ship planning step if controlled by remote actor
                    ship_obj.plan(t=self.t, dt=self.dt, do_list=tracks, enc=self.enc, w=disturbance_data)

                if i > 0 and self.determine_ship_grounding(i):  # Make grounded obstacle ships stop
                    ship_obj.set_references(np.zeros((9, 1)))

            sim_data_dict[f"Ship{i}"] = ship_obj.get_sim_data(self.t, self.timestamp_start)
            sim_data_dict[f"Ship{i}"]["sensor_measurements"] = self.recent_sensor_measurements[i]
            sim_data_dict[f"Ship{i}"]["colav"] = ship_obj.get_colav_data()

            if ship_obj.t_start <= self.t:
                ship_obj.forward(self.dt, disturbance_data)

        self.t += self.dt
        return sim_data_dict

    def distance_to_nearby_vessels(self, ship_idx: int = 0) -> np.ndarray:
        """Calculates the distance to nearby vessels for a ship.

        Args:
            ship_idx (int, optional): Index of the ship to calculate the distance to nearby vessels for. Defaults to 0.

        Returns:
            np.ndarray: Array containing the distances to nearby vessels.
        """
        ship_state = self.ship_list[ship_idx].csog_state
        distances = []
        for i, other_ship_obj in enumerate(self.ship_list):
            if i == ship_idx:
                continue
            if other_ship_obj.t_start <= self.t:
                other_ship_state = other_ship_obj.csog_state
                distances.append(np.linalg.norm(ship_state[:2] - other_ship_state[:2]))
            else:
                distances.append(1e12)
        assert len(distances) == len(self.ship_list) - 1
        return np.array(distances)

    def determine_ship_collision(self, ship_idx: int = 0) -> bool:
        """Determines whether a ship is in a collision state.

        Args:
            ship_idx (int, optional): Index of the ship to check for collision. Defaults to 0.

        Returns:
            bool: True if the ship is in a collision state, False otherwise.
        """
        distances = self.distance_to_nearby_vessels(ship_idx)
        other_ship_list = [other_ship_obj for i, other_ship_obj in enumerate(self.ship_list) if i != ship_idx]
        for i, other_ship_obj in enumerate(other_ship_list):
            if distances[i] <= self.ship_list[ship_idx].length / 2.0:  # + other_ship_obj.length / 2.0:
                return True
        return False

    def distance_to_grounding(self, ship_idx: int = 0) -> float:
        """Calculates the distance to grounding for a ship.

        Args:
            ship_idx (int, optional): Index of the ship to calculate the distance to grounding for. Defaults to 0.

        Returns:
            float: Distance to grounding for the ship.
        """
        ship_state = self.ship_list[ship_idx].csog_state
        d2land = mapf.min_distance_to_hazards(self.relevant_grounding_hazards, ship_state[1], ship_state[0])
        return d2land

    def determine_ship_grounding(self, ship_idx: int = 0) -> bool:
        """Determines whether a ship is in a grounding state.

        Args:
            ship_idx (int, optional): Index of the ship to check for grounding. Defaults to 0.

        Returns:
            bool: True if the ship is in a grounding state, False otherwise.
        """
        d2grounding = self.distance_to_grounding(ship_idx)
        return d2grounding <= self.ship_list[ship_idx].length

    def determine_ship_goal_reached(self, ship_idx: int = 0) -> bool:
        """Determines whether the ship has reached its goal.

        Returns:
            bool: True if the own-ship has reached its goal, False otherwise.
        """
        if self.ship_list[ship_idx].goal_csog_state.size > 0:
            goal_state = self.ship_list[ship_idx].goal_csog_state
        elif self.ship_list[ship_idx].waypoints.size > 1:
            goal_state = self.ownship.waypoints[:, -1]
        else:
            raise ValueError(
                "Either the goal pose must be provided, or a sufficient number of waypoints for the ship to follow!"
            )
        ship_state = self.ship_list[ship_idx].csog_state
        d2goal = np.linalg.norm(ship_state[:2] - goal_state[:2])
        scale_factor = 4.0
        return d2goal <= self.ship_list[ship_idx].length * scale_factor


def extract_valid_sensor_measurements(t: float, recent_sensor_measurements: list, sensor_measurements_i: list) -> list:
    """Extracts non-NaN sensor measurements from the recent sensor measurements list and appends them to the most recent sensor measurements list.

    Args:
        t (float): Current simulation time.
        recent_sensor_measurements (list): List of most recent valid (non-nan) sensor measurements for the current ship
        sensor_measurements_i (list): List of new sensor measurements for the current ship.

    Returns:
        list: List of updated most recent valid (non-nan) sensor measurements for the current ship
    """
    if t == 0.0:
        recent_sensor_measurements = sensor_measurements_i
    for _, meas in enumerate(sensor_measurements_i):
        if not meas:
            continue
        if np.isnan(meas).any():
            continue
        recent_sensor_measurements.append(meas)
    return recent_sensor_measurements
