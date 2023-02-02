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
    ais_data_column_format: list

    scenario_generator: sm.Config
    visualizer: viz.Config

    @classmethod
    def from_dict(cls, config_dict: dict):
        simulator = config_dict["simulator"]
        config = Config(
            scenario_files=simulator["scenario_files"],
            save_scenario_results=simulator["save_scenario_results"],
            verbose=simulator["verbose"],
            ais_data_column_format=simulator["ais_data_column_format"],
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

    def generate_scenarios_from_config(self) -> list:
        """Generates scenarios listed in the simulator config.

        Returns:
            dict: Dictionary containing list of ship objects, scenario configuration objects and relevant ENC objects for each scenario.
        """

        scenario_list = []
        for i, scenario_file in enumerate(self._config.scenario_files):
            scenario_data = {}
            if self._config.verbose:
                print("\rScenario generator: Creating scenario nr {i}: {scenario_file}...")
            ship_list, scenario_config, scenario_enc = self._scenario_generator.generate(dp.scenarios / scenario_file)
            if self._config.verbose:
                print("\rScenario generator: Finished creating scenario nr {i}: {scenario_file}.")

            scenario_data["ship_list"] = ship_list
            scenario_data["scenario_config"] = scenario_config
            scenario_data["scenario_enc"] = scenario_enc
            scenario_list.append(scenario_data)
        return scenario_list

    def run(self, scenario_list: Optional[list] = None) -> dict:
        """Runs through all specified scenarios. If none are specified, the scenarios are generated from the config file and run through.

        Args:
            input_scenarios (Optional[list]): Premade list of created/configured scenarios. Each entry contains a list of ship objects, scenario configuration objects and relevant ENC objects. Defaults to None.

        Returns:
            dict: Dictionary containing list of simulation data, simulated AIS data, ship info and vessel data (for evaluation) for each scenario.
        """
        if self._config.verbose:
            print("\rSimulator: Started running through scenarios...")

        sim_data_list = []
        ais_data_list = []
        ship_info_list = []
        vessels_data_list = []
        scenario_config_list = []
        scenario_enc_list = []

        if scenario_list is None:
            scenario_list = self.generate_scenarios_from_config()

        for i, scenario_data in enumerate(scenario_list):
            ship_list = scenario_data["ship_list"]
            scenario_config = scenario_data["scenario_config"]
            scenario_enc = scenario_data["scenario_enc"]
            scenario_file = scenario_config.filename

            if self._config.verbose:
                print(f"\rSimulator: Running scenario nr {i}: {scenario_file}...")
            sim_data, ais_data, ship_info, sim_times = self.run_scenario(ship_list, scenario_config, scenario_enc)
            if self._config.verbose:
                print("\rSimulator: Finished running through scenario nr {i}: {scenario_file}.")

            self._visualizer.visualize_results(
                self._scenario_generator.enc,
                ship_list,
                sim_data,
                sim_times,
                save_figs=True,
                save_file_path=dp.figure_output / scenario_config.name,
            )

            vessel_data = mhm.convert_simulation_data_to_vessel_data(sim_data, ship_info, scenario_config.utm_zone)
            vessels_data_list.append(vessel_data)
            sim_data_list.append(sim_data)
            ais_data_list.append(ais_data)
            ship_info_list.append(ship_info)
            scenario_config_list.append(scenario_config)
            scenario_enc_list.append(scenario_enc)

        output = {}
        output["sim_data_list"] = sim_data_list
        output["ais_data_list"] = ais_data_list
        output["ship_info_list"] = ship_info_list
        output["vessels_data_list"] = vessels_data_list
        output["scenario_config_list"] = scenario_config_list
        output["scenario_enc_list"] = scenario_enc_list
        if self._config.verbose:
            print("\rSimulator: Finished running through scenarios.")
        return output

    def run_scenario(self, ship_list: list, scenario_config: sm.ScenarioConfig, scenario_enc: senc.ENC) -> Tuple[pd.DataFrame, pd.DataFrame, dict, np.ndarray]:
        """Runs the simulator for a scenario specified by the ship object array, using a time step dt_sim.

        Args:
            ship_list (list): 1 x n_ships array of configured Ship objects. Each ship
            is assumed to be properly configured and initialized to its initial state at
            the scenario start (t0).
            scenario_config (ScenarioConfig): Scenario configuration object.
            scenario_enc (senc.ENC): ENC object relevant for the scenario.


        Returns: a tuple containing:
            sim_data (DataFrame): Dataframe/table containing the ship simulation data.
            ais_data (DataFrame): Dataframe/table containing the AIS data broadcasted from all ships.
            ship_info (dict): Dictionary containing the ship info for each ship.
            sim_times (np.array): Array containing the simulation times.
        """
        self._visualizer.init_live_plot(scenario_enc, ship_list)

        sim_data = []
        ais_data = []
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

                sim_data_dict[f"Ship{i}"] = ship_obj.get_ship_sim_data(t, timestamp_start)
                sim_data_dict[f"Ship{i}"]["sensor_measurements"] = most_recent_sensor_measurements[i]

                if t % 1.0 / ship_obj.ais_msg_freq == 0:
                    ais_data_row = ship_obj.get_ais_data(int(t), timestamp_start, scenario_config.utm_zone)
                    ais_data.append(ais_data_row)

            sim_data.append(sim_data_dict)

            if t % 10.0 < 0.0001:
                self._visualizer.update_live_plot(t, self._scenario_generator.enc, ship_list, most_recent_sensor_measurements[0])

        return pd.DataFrame(sim_data), pd.DataFrame(ais_data, columns=self._config.ais_data_column_format), ship_info, sim_times


def extract_valid_sensor_measurements(t: float, most_recent_sensor_measurements: list, sensor_measurements_i: list) -> list:
    if t == 0.0:
        most_recent_sensor_measurements = sensor_measurements_i
    for _, meas in enumerate(sensor_measurements_i):
        if not meas:
            continue
        if np.isnan(meas).any():
            continue
        most_recent_sensor_measurements.append(meas)
    return most_recent_sensor_measurements
