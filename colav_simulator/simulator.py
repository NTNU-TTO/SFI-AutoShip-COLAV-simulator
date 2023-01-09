"""
    simulator.py

    Summary:
        Contains class definitions for the simulator, enables the
        simulation of a diverse set of COLAV scenarios from files.

    Author: Trym Tengesdal, Magne Aune, Joachim Miller
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp
import colav_simulator.scenario_management as sm
import numpy as np
import pandas as pd
from colav_simulator.viz.visualizer import Visualizer

np.set_printoptions(suppress=True, formatter={"float_kind": "{:.4f}".format})


@dataclass
class Config:
    """Simulation related configuration/parameter class."""

    t_start: float
    dt_sim: float
    t_end: float
    scenario_files: list
    save_scenario_results: bool
    visualize: bool
    verbose: bool
    ais_data_column_format: list


class Simulator:
    """Class for simulating collision avoidance/maritime vessel scenarios."""

    _config: Config
    _scenario_generator: sm.ScenarioGenerator
    _visualizer: Visualizer

    def __init__(self, config_file: Path = dp.simulator_config, **kwargs) -> None:
        """Initializes the simulator.

        Additional key-value arguments can be passed to override the settings in the config file:

        Args:
            config_file (Path, optional): _description_. Defaults to dp.simulator_config.
            kwargs: key-value dictionary of settings to override. This includes:
                scenario_files (List[str]): List of scenario files to run.

        """
        self._config = cp.extract(Config, config_file, dp.simulator_schema, **kwargs)

        self._scenario_generator = sm.ScenarioGenerator()

        self._visualizer = Visualizer()

    def run(
        self,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        dt_sim: Optional[float] = None,
    ) -> dict:
        """Runs through all specified scenarios.

        Args:
            t_start (Optional[float]): Simulation start time
            t_end (Optional[float]): Simulation end time
            dt_sim (Optional[float]): Simulation time step


        Returns:
            dict: Dictionary containing list of simulation data, simulated AIS data, ship info and vessel data (for evaluation) for each scenario.
        """
        if self._config.verbose:
            print("Running simulator...")

        if t_start is None:
            t_start = self._config.t_start

        if t_end is None:
            t_end = self._config.t_end

        if dt_sim is None:
            dt_sim = self._config.dt_sim

        sim_data_list = []
        ais_data_list = []
        ship_info_list = []
        vessels_data_list = []

        sim_times = np.arange(t_start, t_end, dt_sim)
        for i, scenario_file in enumerate(self._config.scenario_files):

            ship_list, _, scenario_config = self._scenario_generator.generate(dp.scenarios / scenario_file, sample_interval=dt_sim)

            if self._config.verbose:
                print(f"Running scenario nr {i}: {scenario_file}...")
            sim_data, ais_data, ship_info = self.run_scenario(ship_list, sim_times, scenario_config.utm_zone)

            if self._config.visualize and False:
                self._visualizer.visualize_results(
                    self._scenario_generator.enc,
                    ship_list,
                    sim_data,
                    sim_times,
                    save_figs=True,
                    save_file_path=dp.figure_output / scenario_config.name,
                )

            # TODO: Add saving of scenario viz results if specified in config

            vessel_data = mhm.convert_simulation_data_to_vessel_data(sim_data, ship_info, scenario_config.utm_zone)
            vessels_data_list.append(vessel_data)
            sim_data_list.append(sim_data)
            ais_data_list.append(ais_data)
            ship_info_list.append(ship_info)

        output = {}
        output["sim_data_list"] = sim_data_list
        output["ais_data_list"] = ais_data_list
        output["ship_info_list"] = ship_info_list
        output["vessels_data_list"] = vessels_data_list
        return output

    def run_scenario(self, ship_list: list, sim_times: np.ndarray, utm_zone: int):
        """Runs the simulator for a scenario specified by the ship object array, using a time step dt_sim.

        Args:
            ship_list (list): 1 x n_ships array of configured Ship objects. Each ship
            is assumed to be properly configured and initialized to its initial state at
            the scenario start (t0).
            sim_times (np.ndarray): Array of sim_times to simulate the ships.
            utm_zone (int): UTM zone used for the planar coordinate system.

        Returns:
            sim_data (DataFrame): Dataframe/table containing the ship simulation data.
            ais_data (DataFrame): Dataframe/table containing the AIS data broadcasted from all ships.
            ship_info (dict): Dictionary containing the ship info for each ship.
        """
        if self._config.visualize:
            self._visualizer.init_live_plot(self._scenario_generator.enc, ship_list)

        sim_data = []
        ais_data = []
        ship_info = {}
        for i, ship_obj in enumerate(ship_list):
            ship_info[f"Ship{i}"] = ship_obj.get_ship_info()

        timestamp_start = mhm.current_utc_timestamp()
        t_prev = sim_times[0]
        for _, t in enumerate(sim_times):
            dt_sim = t - t_prev
            t_prev = t

            sim_data_dict = {}
            sensor_measurements = []
            true_do_states = []
            for i, ship_obj in enumerate(ship_list):
                if t == 0.0:
                    print(f"Ship {i} starts at {ship_obj.t_start} | t is now {t}")
                if ship_obj.t_start <= t:
                    state = mhm.convert_sog_cog_state_to_vxvy_state(ship_obj.pose)
                    true_do_states.append((i, state))

            for i, ship_obj in enumerate(ship_list):
                if i == 0:
                    relevant_true_do_states = mhm.get_relevant_do_states(true_do_states, i)
                    _, _, sensor_measurements_i = ship_obj.track_obstacles(t, dt_sim, relevant_true_do_states)
                    sensor_measurements.append(sensor_measurements_i)
                    for sensor_idx, meas in enumerate(sensor_measurements_i):
                        if len(meas) > 0 and ~np.isnan(meas).any():
                            sensor_measurements[i][sensor_idx] = meas

                if dt_sim > 0 and ship_obj.t_start <= t:
                    ship_obj.forward(dt_sim)

                sim_data_dict[f"Ship{i}"] = ship_obj.get_ship_sim_data(int(t), timestamp_start)
                sim_data_dict[f"Ship{i}"]["sensor_measurements"] = sensor_measurements_i

                if t % 1.0 / ship_obj.ais_msg_freq == 0:
                    ais_data_row = ship_obj.get_ais_data(int(t), timestamp_start, utm_zone)
                    ais_data.append(ais_data_row)

            sim_data.append(sim_data_dict)

            if self._config.visualize and t % 10.0 < 0.0001:
                self._visualizer.update_live_plot(t, self._scenario_generator.enc, ship_list, sensor_measurements)

        return pd.DataFrame(sim_data), pd.DataFrame(ais_data, columns=self._config.ais_data_column_format), ship_info
