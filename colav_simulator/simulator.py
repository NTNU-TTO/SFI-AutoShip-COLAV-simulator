"""
    simulator.py

    Summary:
        Contains class definitions for the simulator, enables the
        simulation of a diverse set of COLAV scenarios from files.

    Author: Trym Tengesdal, Magne Aune, Joachim Miller
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp
import colav_simulator.scenario_management as sm
import colav_simulator.viz.visualizer as viz
import numpy as np
import pandas as pd
import seacharts.enc as senc

np.set_printoptions(suppress=True, formatter={"float_kind": "{:.4f}".format})


@dataclass
class Config:
    """Simulation related configuration/parameter class."""

    scenario_files: list
    save_scenario_results: bool
    verbose: bool

    scenario_generator: sm.Config
    visualizer: viz.Config

    @classmethod
    def from_dict(cls, config_dict: dict):
        simulator = config_dict["simulator"]
        config = Config(
            scenario_files=simulator["scenario_files"],
            save_scenario_results=simulator["save_scenario_results"],
            verbose=simulator["verbose"],
            scenario_generator=sm.Config.from_dict(config_dict["scenario_generator"]),
            visualizer=viz.Config.from_dict(config_dict["visualizer"]),
        )

        return config


class Simulator:
    """Class for simulating collision avoidance/maritime vessel scenarios."""

    _config: Config
    _scenario_generator: sm.ScenarioGenerator
    _visualizer: viz.Visualizer

    def __init__(self, config_file: Path = dp.simulator_config, **kwargs) -> None:
        """Initializes the simulator.

        Additional key-value arguments can be passed to override the settings in the config file:

        Args:
            config_file (Path, optional): _description_. Defaults to dp.simulator_config.
            kwargs: key-value dictionary of settings to override. This includes:
                scenario_files (List[str]): List of scenario files to run.

        """
        self._config = cp.extract(Config, config_file, dp.simulator_schema, **kwargs)

        self._scenario_generator = sm.ScenarioGenerator(self._config.scenario_generator)

        self._visualizer = viz.Visualizer(self._config.visualizer)

    def run(self, scenario_data_list: Optional[list] = None) -> list:
        """Runs through all specified scenarios with their number of episodes. If none are specified, the scenarios are generated from the config file and run through.

        Args:
            scenario_data_list (Optional[list]): Premade list of created/configured scenarios. Each entry contains a list of ship objects, scenario configuration objects and relevant ENC objects. Defaults to None.

        Returns:
            list: List of dictionaries containing the following simulation data for each scenario:
            - episode_simdata_list (list): List of dictionaries containing the following simulation data for each scenario episode:
                - sim_data (pd.DataFrame): Dataframe containing the simulated data for each ship.
                - ais_data (pd.DataFrame): Dataframe containing the simulated ais data for each ship.
                - ship_info (pd.DataFrame): Dataframe containing the ship info for each ship.
                - config (sm.ScenarioConfig): Configuration object for the scenario episode.
            - enc (senc.Enc): ENC object used in all the scenario episodes.
        """
        if scenario_data_list is None:
            files = [dp.scenarios / f for f in self._config.scenario_files]
            # scenario_data_list = self._scenario_generator.generate_scenarios_from_files(files, self._config.verbose)
            scenario_data = self._scenario_generator.load_scenario_from_folder(dp.scenarios / "saved", "rogaland_random", self._config.verbose)
            scenario_data_list = [scenario_data]

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
                episode_config = episode_data["config"]
                scenario_episode_file = episode_config.filename

                if self._config.verbose:
                    print(f"\rSimulator: Running scenario episode nr {ep + 1}: {scenario_episode_file}...")
                sim_data, ship_info, sim_times = self.run_scenario_episode(ship_list, episode_config, scenario_enc)
                if self._config.verbose:
                    print(f"\rSimulator: Finished running through scenario episode nr {ep + 1}: {scenario_episode_file}.")

                self._visualizer.visualize_results(
                    self._scenario_generator.enc,
                    ship_list,
                    sim_data,
                    sim_times,
                    save_figs=True,
                    save_file_path=dp.figure_output / episode_config.name,
                )

                vessel_data = mhm.convert_simulation_data_to_vessel_data(sim_data, ship_info, episode_config.utm_zone)

                episode_simdata["vessel_data"] = vessel_data
                episode_simdata["sim_data"] = sim_data
                episode_simdata["ship_info"] = ship_info
                episode_simdata["config"] = episode_config
                episode_simdata_list.append(episode_simdata)

            if self._config.verbose:
                print(f"\rSimulator: Finished running through episodes of scenario nr {i}.")

            scenario_simdata["episode_simdata_list"] = episode_simdata_list
            scenario_simdata["enc"] = scenario_enc
            scenario_simdata_list.append(scenario_simdata)

        if self._config.verbose:
            print("\rSimulator: Finished running through scenarios.")
        return scenario_simdata_list

    def run_scenario_episode(self, ship_list: list, scenario_config: sm.ScenarioConfig, scenario_enc: senc.ENC) -> Tuple[pd.DataFrame, dict, np.ndarray]:
        """Runs the simulator for a scenario episode specified by the ship object array, using a time step dt_sim.

        Args:
            ship_list (list): 1 x n_ships array of configured Ship objects. Each ship
            is assumed to be properly configured and initialized to its initial state at
            the scenario start (t0).
            scenario_config (ScenarioConfig): Scenario episode configuration object.
            scenario_enc (senc.ENC): ENC object relevant for the scenario.


        Returns: a tuple containing:
            sim_data (DataFrame): Dataframe/table containing the ship simulation data.
            ship_info (dict): Dictionary containing the ship info for each ship.
            sim_times (np.array): Array containing the simulation times.
        """
        self._visualizer.init_live_plot(scenario_enc, ship_list)

        sim_data = []
        ship_info = {}
        for i, ship_obj in enumerate(ship_list):
            ship_info[f"Ship{i}"] = ship_obj.get_ship_info()

        timestamp_start = mhm.current_utc_timestamp()
        most_recent_sensor_measurements = [None] * len(ship_list)
        sim_times = np.arange(scenario_config.t_start, scenario_config.t_end, scenario_config.dt_sim)
        t_prev = sim_times[0]
        for _, t in enumerate(sim_times):
            dt_sim = t - t_prev
            t_prev = t

            sim_data_dict = {}
            true_do_states = []
            for i, ship_obj in enumerate(ship_list):
                if t == 0.0:
                    print(f"Ship {i} starts at {ship_obj.t_start}")
                if ship_obj.t_start <= t:
                    vxvy_state = mhm.convert_csog_state_to_vxvy_state(ship_obj.csog_state)
                    true_do_states.append((i, vxvy_state))

            for i, ship_obj in enumerate(ship_list):
                relevant_true_do_states = mhm.get_relevant_do_states(true_do_states, i)
                tracks, sensor_measurements_i = ship_obj.track_obstacles(t, dt_sim, relevant_true_do_states)

                most_recent_sensor_measurements[i] = extract_valid_sensor_measurements(t, most_recent_sensor_measurements[i], sensor_measurements_i)

                if dt_sim > 0 and ship_obj.t_start <= t:
                    ship_obj.plan(
                        t=t,
                        dt=dt_sim,
                        do_list=tracks,
                        enc=scenario_enc,
                    )
                    ship_obj.forward(dt_sim)

                sim_data_dict[f"Ship{i}"] = ship_obj.get_sim_data(t, timestamp_start)
                sim_data_dict[f"Ship{i}"]["sensor_measurements"] = most_recent_sensor_measurements[i]

            sim_data.append(sim_data_dict)

            if t % 10.0 < 0.0001:
                self._visualizer.update_live_plot(t, self._scenario_generator.enc, ship_list, most_recent_sensor_measurements[0])

        return pd.DataFrame(sim_data), ship_info, sim_times


def extract_valid_sensor_measurements(t: float, most_recent_sensor_measurements: list, sensor_measurements_i: list) -> list:
    """Extracts non-NaN sensor measurements from the recent sensor measurements list and appends them to the most recent sensor measurements list.

    Args:
        t (float): Current simulation time.
        most_recent_sensor_measurements (list): List of most recent valid (non-nan) sensor measurements for the current ship
        sensor_measurements_i (list): List of new sensor measurements for the current ship.

    Returns:
        list: List of updated most recent valid (non-nan) sensor measurements for the current ship
    """
    if t == 0.0:
        most_recent_sensor_measurements = sensor_measurements_i
    for _, meas in enumerate(sensor_measurements_i):
        if not meas:
            continue
        if np.isnan(meas).any():
            continue
        most_recent_sensor_measurements.append(meas)
    return most_recent_sensor_measurements
