"""
    simulator.py

    Summary:
        Contains class definitions for the simulator, enables the
        simulation of a diverse set of COLAV scenarios from files.

    Author: Trym Tengesdal, Magne Aune, Joachim Miller
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import colav_simulator.common.config_parsing as config_parsing
import colav_simulator.common.paths as dp  # default paths
import numpy as np
import pandas as pd
from colav_simulator.scenario_generator import ScenarioGenerator
from colav_simulator.viz.visualizer import Visualizer
from yaspin import yaspin

np.set_printoptions(suppress=True, formatter={"float_kind": "{:.2f}".format})


@dataclass
class Config:
    """Simulation related configuration/parameter class."""

    t_start: float
    dt_sim: float
    t_end: float
    scenario_files: list
    save_animation: bool
    show_animation: bool
    show_waypoints: bool
    verbose: bool
    ais_data_column_format: list


class Simulator:
    """Class for simulating collision avoidance/maritime vessel scenarios."""

    _config: Config
    _scenario_generator: ScenarioGenerator
    _visualizer: Visualizer

    def __init__(self, config_file: Path = dp.simulator_config, **kwargs) -> None:
        """Initializes the simulator.

        Additional key-value arguments can be passed to override the settings in the config file:

        Args:
            config_file (Path, optional): _description_. Defaults to dp.simulator_config.
            kwargs: key-value dictionary of settings to override. This includes:
                scenario_files (List[str]): List of scenario files to run.

        """

        self._config = config_parsing.extract(Config, config_file, dp.simulator_schema, **kwargs)

        self._scenario_generator = ScenarioGenerator()

        self._visualizer = Visualizer(enc=self._scenario_generator.enc)

        # override

        # => save newly created scenarios to file if requested.
        # => run the scenarios and save sim_data and emulated ais_data to file. Toggle animation on/off based on config.
        # => The COLAV evaluator can then load the sim & ais data from file and plot/evaluate the results.

    def run(
        self,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        dt_sim: Optional[float] = None,
        save_results: bool = False,
    ):
        """Runs through all specified scenarios.

        Args:
            t_start (Optional[float]): Simulation start time
            t_end (Optional[float]): Simulation end time
            dt_sim (Optional[float]): Simulation time step


        Returns:
            Tuple[list, list]: Lists of simulation data and AIS data for each scenario.
        """

        if t_start is None:
            t_start = self._config.t_start

        if t_end is None:
            t_end = self._config.t_end

        if dt_sim is None:
            dt_sim = self._config.dt_sim

        sim_data_list = []
        ais_data_list = []

        sim_times = np.arange(t_start, t_end, dt_sim)
        for scenario_file in self._config.scenario_files:
            if "new" in scenario_file or "new_scenario" in scenario_file:
                ship_list = self._scenario_generator.generate()
            else:
                "pass"
                # ship_list = load_scenario_definition(Path(scenario_file))

            sim_data, ais_data = self.run_scenario(ship_list, sim_times)

            self._visualizer.visualize(
                ship_list=ship_list,
                data=sim_data,
                times=sim_times,
                save_path=dp.animation_output / (scenario_file + ".gif"),
            )

            sim_data_list.append(sim_data)
            ais_data_list.append(ais_data)

        return sim_data_list, ais_data_list

    def run_scenario(self, ship_list: list, sim_times: np.ndarray):
        """Runs the simulator for a scenario specified by the ship object array, using a time step dt_sim.

        Args:
            ships (list): 1 x n_ships array of configured ship objects. Each ship
            is assumed to be properly configured and initialized to its initial state at
            the scenario start (t0).
            sim_times (np.ndarray): 1 x n_samples array of sim_times to simulate the ships.

        Returns:
            sim_data (DataFrame): Dataframe/table containing the ship simulation data.
            ais_data (DataFrame): Dataframe/table containing the AIS data broadcasted from all ships.
        """
        sim_data = []
        ais_data = []
        t0 = sim_times[0]
        t_prev = t0
        for _, t in enumerate(sim_times):
            dt_sim = t - t_prev
            t_prev = t

            sim_data_dict = {}
            for i, ship in enumerate(ship_list):
                if dt_sim > 0:
                    ship.track_obstacles()

                    ship.forward(dt_sim)

                sim_data_dict[f"Ship{i}"] = ship.get_ship_nav_data(t)

                if t % 1.0 / ship.ais_msg_freq == 0:
                    ais_data_row = ship.get_ais_data(int(t))
                    ais_data.append(ais_data_row)

            sim_data.append(sim_data_dict)

        return pd.DataFrame(sim_data), pd.DataFrame(ais_data, columns=self._config.ais_data_column_format)


def load_scenario_definition(loadfile: Path):
    """
    Loads a scenario definition from a json file and processes into a list of poses,
    waypoints and speed plans (the definition).

    Parameters:
        loadfile (pathlib.Path): Absolute path to scenario_file.

    Returns:
        dict: Dictionary containing the scenario definition, with keys:
                    poses[i]: pose [x, y, U, psi] for ship i
                    waypoint_list[i]: waypoints for ship i
                    speed_plans[i]: speed_plan for ship i
    """

    with loadfile.open(mode="r") as file:
        data = json.load(file)

    pose_list = data["poses"]
    waypoint_list = []  # data['waypoints']
    speed_plan_list = data["speed_plans"]

    for i in range(len(data["waypoints"])):
        waypoints = [tuple(row) for row in data["waypoints"][i]]
        waypoint_list.append(waypoints)

    return pose_list, waypoint_list, speed_plan_list


def save_scenario(
    pose_list: List[np.ndarray], waypoint_list: List[np.ndarray], speed_plan_list: List[np.ndarray], savefile: Path
):
    """Saves the the scenario defined by the list of ship poses, waypoints and speed plans, to a json file as a dict at savefile
    dict keys:
        pose_list[i]: pose [x,y,u,psi] for ship i
        waypoint_list[i]: waypoints for ship i
        speed_plan_list[i]: speed_plan for ship i
    """

    data = {"poses": pose_list, "waypoints": waypoint_list, "speed_plans": speed_plan_list}
    with savefile.open(mode="w") as file:
        json.dump(data, file, indent=2)
