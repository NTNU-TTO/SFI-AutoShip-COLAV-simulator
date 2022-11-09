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
import yaspin
from colav_simulator.scenario_generator import ScenarioGenerator
from colav_simulator.ships.ship import Config as ShipConfig
from colav_simulator.ships.ship import Ship

np.set_printoptions(suppress=True, formatter={"float_kind": "{:.2f}".format})


@dataclass
class Config:
    """Simulation related configuration/parameter class."""

    t_start: float
    dt_sim: float
    t_end: float
    scenario_files: List[str]
    run_all_scenarios: bool
    new_scenario: bool
    save_animation: bool
    show_animation: bool
    show_waypoints: bool
    verbose: bool
    ais_data_column_format: List[str]

simulator_converter = {
    float: float,
    float: float,
    float: float,
    list: list,
    bool: bool,
    bool: bool,
    bool: bool,
    bool: bool,
    bool: bool,
    bool: bool,
    list: list,
}

class Simulator:
    """Class for simulating collision avoidance/maritime vessel scenarios."""

    _config: Config
    _scenario_generator: ScenarioGenerator

    def __init__(self, scenario_files: List[Path], config_file: Path = dp.simulator_config) -> None:

        self._config = config_parsing.extract(Config, config_file, dp.simulator_schema, simulator_converter)

        # => then load relevant scenarios if provided, or create new ones.
        # => save newly created scenarios to file if requested.
        # => run the scenarios and save sim_data and emulated ais_data to file. Toggle animation on/off based on config.
        # => The COLAV evaluator can then load the sim & ais data from file and plot/evaluate the results.

    @yaspin(text="Running simulator...")
    def run(
        self, t_start: Optional[float], t_end: Optional[float], dt_sim: Optional[float], save_results: bool = False
    ):
        """Runs through all specified scenarios.

        Args:
            t_start (Optional[float]): Simulation start time
            t_end (Optional[float]): Simulation end time
            dt_sim (Optional[float]): Simulation time step


        Returns:

        """

        if t_start is None:
            t_start = self._config.t_start

        if t_end is None:
            t_end = self._config.t_end

        if dt_sim is None:
            dt_sim = self._config.dt_sim

        n_scenarios = len(self._config.scenario_files)

        sim_times = np.arange(t_start, t_end, dt_sim)
        for scenario_file in self._config.scenario_files:
            if "new" in scenario_file or "new_scenario" in scenario_file:
                pose_list, waypoints_list, speed_plan_list = self._scenario_generator.generate()
            else:
                pose_list, waypoints_list, speed_plan_list = load_scenario_definition(Path(scenario_file))

            ship_list = configure_ships_in_scenario(pose_list, waypoints_list, speed_plan_list)

            sim_data, ais_data = self.run_scenario(ship_list, sim_times)

        return sim_data_list, ais_data_list

    @yaspin(text="Running scenario...")
    def run_scenario(self, ship_list: List[Ship], sim_times: np.ndarray):
        """Runs the simulator for a scenario specified by the ship object array, using a time step dt_sim.

        Args:
            ships (List[Ship]): 1 x n_ships array of configured ship objects. Each ship
            is assumed to be properly configured and initialized to its initial state at
            the scenario start (t0).
            sim_times (np.ndarray): 1 x n_samples array of sim_times to simulate the ships.

        Returns:
            sim_data (dict): Dictionary containing the simulation data.
            ais_data (Dataframe): Dataframe/table containing the AIS data broadcasted from all ships.
        """
        ais_data = pd.DataFrame(columns=self._config.ais_data_column_format)

        sim_data = {}
        for i, ship in enumerate(ship_list):
            x_i = np.zeros(len(sim_times))
            y_i = np.zeros(len(sim_times))
            U_i_t = np.zeros(len(sim_times))
            chi_i_t = np.zeros(len(sim_times))
            waypoints = np.zeros(len(sim_times))
            sim_data[f"Ship{i}"] = [x_i, y_i, U_i_t, chi_i_t, waypoints]
            sim_data[f"Ship{i}"][4] = ship.waypoints

        t0 = sim_times[0]
        t_prev = t0
        for t_idx, t in enumerate(sim_times, start=1):
            for i, ship in enumerate(ship_list):
                dt_sim = t - t_prev

                ship.track_obstacles()

                ship.forward(dt_sim)

                # Logging
                sim_data[f"Ship{i}"][t_idx] = [*ship.pose]

                if t % 1.0 / ship.ais_msg_freq == 0:
                    ais_data = ais_data.append(
                        ship.get_ais_data(t),
                        ignore_index=True,
                    )
        return sim_data, ais_data


@yaspin(text="Loading scenario...")
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


@yaspin(text="Saving...")
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


@yaspin(text="Configuring ships...")
def configure_ships_from_file(ship_config_file: Path = dp.ship_config):
    """Configures the subsystems of the ships in a scenario from file.

    Parameters:
        ship_config_file (pathlib.Path): Absolute path to the ship configuration file.

    Returns:
        List[Ship]: list of initialized and configured Ship objects.
    """

    ship_list = []
    for i, range(n_ships):
        ship_config = config_parsing.extract(ShipConfig, ship_config_file, dp.ship_schema, converters.ship)
        ship = Ship(
            mmsi=i + 1,
            config=ship_config,
        )
        ship_list.append(ship)

    return ship_list
