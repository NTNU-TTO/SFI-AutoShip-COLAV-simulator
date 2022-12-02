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
import colav_simulator.common.math_functions as mf
import colav_simulator.common.paths as dp  # default paths
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
    save_animation: bool
    save_new_scenarios: bool
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

        self._visualizer = Visualizer(enc=self._scenario_generator.enc)

    def run(
        self,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        dt_sim: Optional[float] = None,
        save_new_scenarios: Optional[bool] = None,
    ):
        """Runs through all specified scenarios.

        Args:
            t_start (Optional[float]): Simulation start time
            t_end (Optional[float]): Simulation end time
            dt_sim (Optional[float]): Simulation time step


        Returns:
            Tuple[list, list]: Lists of simulation data and AIS data for each scenario.
        """
        if self._config.verbose:
            print("Running simulator...")

        if t_start is None:
            t_start = self._config.t_start

        if t_end is None:
            t_end = self._config.t_end

        if dt_sim is None:
            dt_sim = self._config.dt_sim

        if save_new_scenarios is None:
            save_new_scenarios = self._config.save_new_scenarios

        sim_data_list = []
        ais_data_list = []

        sim_times = np.arange(t_start, t_end, dt_sim)
        for i, scenario_file in enumerate(self._config.scenario_files):
            if "new_scenario" in scenario_file:
                ship_list, ship_config_list = self._scenario_generator.generate()

                if save_new_scenarios:
                    sm.save_scenario(ship_config_list, dp.scenarios / (scenario_file + str(i + 1) + ".yaml"))

            else:
                ship_list = sm.load_scenario_definition(dp.scenarios / scenario_file)

            if self._config.verbose:
                print(f"Running scenario nr {i}: {scenario_file}...")
            sim_data, ais_data = self.run_scenario(ship_list, sim_times)

            sim_data_list.append(sim_data)
            ais_data_list.append(ais_data)

        return sim_data_list, ais_data_list

    def run_scenario(self, ship_list: list, sim_times: np.ndarray):
        """Runs the simulator for a scenario specified by the ship object array, using a time step dt_sim.

        Args:
            ship_list (list): 1 x n_ships array of configured Ship objects. Each ship
            is assumed to be properly configured and initialized to its initial state at
            the scenario start (t0).
            sim_times (np.ndarray): 1 x n_samples array of sim_times to simulate the ships.

        Returns:
            sim_data (DataFrame): Dataframe/table containing the ship simulation data.
            ais_data (DataFrame): Dataframe/table containing the AIS data broadcasted from all ships.
        """
        if self._config.visualize:
            self._visualizer.init_live_plot(ship_list)

        sim_data = []
        ais_data = []
        t0 = sim_times[0]
        t_prev = t0
        for _, t in enumerate(sim_times):
            dt_sim = t - t_prev
            t_prev = t

            sim_data_dict = {}
            sensor_measurements = []
            true_do_states = []
            for i, ship_obj in enumerate(ship_list):
                state = mf.convert_sog_cog_state_to_vxvy_state(ship_obj.pose)
                true_do_states.append(state)

            for i, ship_obj in enumerate(ship_list):

                _, _, sensor_measurements_i = ship_obj.track_obstacles(
                    t, dt_sim, mf.get_list_except_element_idx(true_do_states, i)
                )
                sensor_measurements.append(sensor_measurements_i)

                if dt_sim > 0:
                    ship_obj.forward(dt_sim)

                sim_data_dict[f"Ship{i}"] = ship_obj.get_ship_sim_data(t)
                sim_data_dict[f"Ship{i}"]["sensor_measurements"] = sensor_measurements_i

                if t % 1.0 / ship_obj.ais_msg_freq == 0:
                    ais_data_row = ship_obj.get_ais_data(int(t))
                    ais_data.append(ais_data_row)

            sim_data.append(sim_data_dict)

            if self._config.visualize and t % 2.0 < 0.0001:
                self._visualizer.update_live_plot(ship_list, sensor_measurements)

        return pd.DataFrame(sim_data), pd.DataFrame(ais_data, columns=self._config.ais_data_column_format)
