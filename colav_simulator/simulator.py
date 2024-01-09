"""
    simulator.py

    Summary:
        Contains class definitions for the simulator, enables the
        simulation of a diverse set of COLAV scenarios from their definitions.

    Author: Trym Tengesdal, Magne Aune, Joachim Miller
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp
import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.core.stochasticity as stochasticity
import colav_simulator.scenario_management as sm
import colav_simulator.viz.visualizer as viz
import numpy as np
import pandas as pd
import seacharts.enc as senc
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


class Simulator:
    """Class for simulating collision avoidance/maritime vessel scenarios."""

    config: Config
    visualizer: viz.Visualizer

    ownship: Ship
    relevant_grounding_hazards: list  # Grounding hazards relevant for the own-ship
    ship_list: list
    disturbance: Optional[stochasticity.Disturbance]
    enc: senc.ENC
    sconfig: sm.ScenarioConfig
    recent_sensor_measurements: list

    t: float
    t_start: float
    t_end: float
    dt: float
    timestamp_start: int

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

        self.visualizer = viz.Visualizer(self._config.visualizer)

    def toggle_liveplot_visibility(self, visible: bool) -> None:
        """Toggles the visibility of the live plot.

        Args:
            visible (bool): Whether the live plot should be visible or not.
        """
        self.visualizer.toggle_liveplot_visibility(visible)

    def initialize_scenario_episode(
        self,
        ship_list: list,
        sconfig: sm.ScenarioConfig,
        enc: senc.ENC,
        disturbance: Optional[stochasticity.Disturbance] = None,
        colav_systems: Optional[list] = None,
    ) -> None:
        """Initializes the simulation through setting relevant internal state objects.

        Args:
            - ship_list (list): 1 x n_ships array of configured Ship objects. Each ship
            is assumed to be properly configured and initialized to its initial state at
            the scenario start (t0).
            - sconfig (ScenarioConfig): Scenario episode configuration object.
            - enc (senc.ENC): ENC object relevant for the scenario.
            - disturbance (Optional[stochasticity.Disturbance]): Disturbance object relevant for the scenario. Defaults to None.
            - colav_systems (Optional[Any | ci.ICOLAV], optional): List of tuples (ship ID, COLAV system) to use for the selected ships involved in the scenario, overrides the existing ones. Defaults to None.
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

        self.timestamp_start = mhm.current_utc_timestamp()
        self.t = sconfig.t_start
        self.t_start = sconfig.t_start
        self.t_end = sconfig.t_end
        self.dt = sconfig.dt_sim
        self.recent_sensor_measurements: list = [None] * len(self.ship_list)

    def run(self, scenario_data_list: list, ownship_colav_system: Optional[Any | ci.ICOLAV] = None) -> list:
        """Runs through all specified scenarios with their number of episodes. If none are specified, the scenarios are generated from the config file and run through.

        Args:
            - scenario_data_list (list): Premade list of created/configured scenarios. Each entry contains a list of ship objects, scenario configuration objects and relevant ENC objects.
            - ownship_colav_system (Optional[Any | ci.ICOLAV]): COLAV system to use for the ownship, overrides the existing one. Defaults to None.

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
                    ship_list, episode_config, scenario_enc, episode_disturbance, ownship_colav_system
                )

                if self._config.verbose:
                    print(f"\rSimulator: Running scenario episode nr {ep + 1}: {scenario_episode_file}...")
                sim_data, ship_info, sim_times = self.run_scenario_episode()
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

            if self._config.verbose:
                print(f"\rSimulator: Finished running through episodes of scenario nr {i + 1}.")

            scenario_simdata["episode_simdata_list"] = episode_simdata_list
            scenario_simdata["enc"] = scenario_enc
            scenario_simdata_list.append(scenario_simdata)

        if self._config.verbose:
            print("\rSimulator: Finished running through scenarios.")
        return scenario_simdata_list

    def is_terminated(self, verbose: bool = False) -> bool:
        """Check whether the current own-ship state is a terminal state.

        Args:
            verbose (bool): Whether to print out the reason for the termination.

        Returns:
            bool: Whether the current own-ship state is a terminal state
        """
        goal_reached = self.determine_ownship_goal_reached()
        collided = self.determine_ownship_collision()
        grounded = self.determine_ownship_grounding()
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

    def run_scenario_episode(self) -> Tuple[pd.DataFrame, dict, np.ndarray]:
        """Runs the simulator for a scenario episode specified by the ship object array, using a time step dt_sim.

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

            terminated = self.is_terminated(verbose=True)
            truncated = self.is_truncated(verbose=True)
            if terminated or truncated:
                t_end = self.t
                break

        self.visualizer.close_live_plot()

        sim_times = np.arange(self.t_start, t_end, self.dt)
        return pd.DataFrame(sim_data), ship_info, sim_times

    def step(self, remote_actor: bool = False) -> dict:
        """Step through the simulation by one time step, using the specified action for the own-ship.

        Args:
            remote_actor (bool, optional): Whether the own-ship is controlled by a remote actor, i.e. references are set externally from the Ship object. Defaults to False.

        Returns:
            dict: Dictionary containing the current time step simulation data for each ship and the disturbance data if applicable.
        """
        sim_data_dict = {}
        true_do_states = mhm.extract_do_states_from_ship_list(self.t, self.ship_list)

        disturbance_data: Optional[stochasticity.DisturbanceData] = None
        if self.disturbance is not None:
            disturbance_data = self.disturbance.get()
            self.disturbance.update(self.t, self.dt)
            sim_data_dict["currents"] = disturbance_data.currents
            sim_data_dict["wind"] = disturbance_data.wind
            sim_data_dict["waves"] = disturbance_data.waves

        for i, ship_obj in enumerate(self.ship_list):
            relevant_true_do_states = mhm.get_relevant_do_states(true_do_states, i)
            tracks, sensor_measurements_i = ship_obj.track_obstacles(self.t, self.dt, relevant_true_do_states)

            self.recent_sensor_measurements[i] = extract_valid_sensor_measurements(
                self.t, self.recent_sensor_measurements[i], sensor_measurements_i
            )

            # Plans a decision for the ship depending on its configuration
            if ship_obj.t_start <= self.t:
                if not (i == 0 and remote_actor):  # Skip own-ship planning step if controlled by remote actor
                    ship_obj.plan(t=self.t, dt=self.dt, do_list=tracks, enc=self.enc, w=disturbance_data)

            sim_data_dict[f"Ship{i}"] = ship_obj.get_sim_data(self.t, self.timestamp_start)
            sim_data_dict[f"Ship{i}"]["sensor_measurements"] = self.recent_sensor_measurements[i]
            sim_data_dict[f"Ship{i}"]["colav"] = ship_obj.get_colav_data()

            if ship_obj.t_start <= self.t:
                ship_obj.forward(self.dt, disturbance_data)

        self.t += self.dt
        return sim_data_dict

    def determine_ownship_collision(self) -> bool:
        """Determines whether the own-ship is in a collision state.

        Returns:
            bool: True if the own-ship is in a collision state, False otherwise.
        """
        ownship_state = self.ownship.csog_state
        for _, ship_obj in enumerate(self.ship_list[1:]):
            if ship_obj.t_start <= self.t:
                ship_state = ship_obj.csog_state
                d2ship = np.linalg.norm(ownship_state[:2] - ship_state[:2])
                if d2ship <= self.ownship.length / 2.0:
                    return True
        return False

    def determine_ownship_grounding(self) -> bool:
        """Determines whether the own-ship is in a grounding state.

        Returns:
            bool: True if the own-ship is in a grounding state, False otherwise.
        """
        ownship_state = self.ownship.csog_state
        d2land = mapf.min_distance_to_hazards(self.relevant_grounding_hazards, ownship_state[1], ownship_state[0])
        return d2land <= self.ownship.length / 2.0

    def determine_ownship_goal_reached(self) -> bool:
        """Determines whether the own-ship has reached its goal.

        Returns:
            bool: True if the own-ship has reached its goal, False otherwise.
        """
        if self.ownship._goal_state.size > 0:
            goal_state = self.ownship._goal_state
        elif self.ownship._waypoints.size > 1:
            goal_state = self.ownship._waypoints[:, -1]
        else:
            raise ValueError(
                "Either the goal pose must be provided, or a sufficient number of waypoints for the ship to follow!"
            )
        ownship_state = self.ownship.csog_state
        d2goal = np.linalg.norm(ownship_state[:2] - goal_state[:2])
        return d2goal <= self.ownship.length / 2.0


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
