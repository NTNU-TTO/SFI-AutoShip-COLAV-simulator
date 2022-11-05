"""
    simulator.py

    Summary:
        Contains class definitions for the simulator, enables the
        simulation of a diverse set of COLAV scenarios from files.

    Author: Trym Tengesdal, Magne Aune, Melih Akdag, Joachim Miller
"""
import json
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import colav_simulator.ships as ships
import numpy as np
import pandas as pd
from colav_simulator.colav_algs.sbmpc import SBMPC
from colav_simulator.scenario_generator import create_scenario
from colav_simulator.ships.ship import Ship

np.set_printoptions(suppress=True, formatter={"float_kind": "{:.2f}".format})


@dataclass
class Config:
    """Configuration class for managing all parameters/settings related to the simulation."""

    ships: Optional[List[ships.ship.Config]]


class Simulator:
    """Class for simulating collision avoidance/maritime vessel scenarios

    Internal variables:
        dd
    """

    _ais_data_column_format: List[str] = [
        "mmsi",
        "lon",
        "lat",
        "date_time_utc",
        "sog",
        "cog",
        "true_heading",
        "nav_status",
        "message_nr",
        "source",
    ]

    def __init__(self, config_file: Optional[str]) -> None:
        pass

    def run_scenario(self, ships: List[Ship], sim_times: np.ndarray):
        """Runs the simulator for a scenario specified by the ship object array, using a time step dt_sim.

        Args:
            ships (List[Ship]): 1 x n_ships array of configured ship objects. Each ship
                                is assumed to be properly configured and initialized to its
                                initial state at the scenario start (t0).
            sim_times (np.ndarray): 1 x n_samples array of sim_times to simulate the ships.

        Returns:
            sim_data (dict): Dictionary containing the simulation data.
            ais_data (Dataframe): Dataframe/table containing the AIS data broadcasted from all ships.
        """
        ais_data = pd.DataFrame(columns=self._ais_data_column_format)

        sim_data = {}
        for i, ship in enumerate(ships):
            x_i = np.zeros(len(sim_times))
            y_i = np.zeros(len(sim_times))
            x_i_t = np.zeros(len(sim_times))
            y_i_t = np.zeros(len(sim_times))
            waypoints = np.zeros(len(sim_times))
            sim_data[f"Ship{i}"] = [x_i, y_i, x_i_t, y_i_t, waypoints]
            sim_data[f"Ship{i}"][4] = ship.waypoints

        t0 = sim_times[0]
        t_prev = t0
        for t_idx, t in enumerate(sim_times, start=1):
            for i, ship in enumerate(ships):
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


def load_scenario_definition(loadfile):
    """
    Loads the scenario init from a json file as a dict
    dict keys:
        poses[i]: pose [x,y,u,psi] for ship i
        waypoints[i]: waypoints for ship i
        speed_plans[i]: speed_plan for ship i
    """
    data = json.load(open(f"scenarios/{loadfile}"))
    pose_list = data["poses"]
    waypoint_list = []  # data['waypoints']
    speed_plan_list = data["speed_plans"]

    for i in range(len(data["waypoints"])):
        wp = [tuple(row) for row in data["waypoints"][i]]
        waypoint_list.append(wp)
    return pose_list, waypoint_list, speed_plan_list


def init_scenario(scenario_file: Optional[pathlib.Path] = None):
    """Initializes and configures ships in the considered scenario. Their route plans
    (waypoints) and speed plans are loaded from a json file.

    Parameters:
        scenario_file (pathlib.Path, optional): Absolute path to scenario_file. If None,
                                                a new scenario is created based on configuration.

    Returns:
        List[Ship]: list of initialized and configured Ship objects in the scenario.
    """

    if scenario_file:
        # Load scenario definition from file
        pose_list, waypoint_list, speed_plan_list = load_scenario_definition(scenario_file)
        # Get config parameters for each ship
        ship_model_name_list, sensors_list, LOS_params_list = get_ship_parameters(num_ships=len(pose_list))
    else:
        # Get config parameters for each ship
        ship_model_name_list, sensors_list, LOS_params_list = get_ship_parameters(num_ships=num_ships)
        # Create and save scenario definition
        pose_list, waypoint_list, speed_plan_list = create_scenario(
            num_ships, scenario_num, ship_model_name_list, os_max_speed, ts_max_speed, num_waypoints
        )
        save_scenario_definition(pose_list, waypoint_list, speed_plan_list, scenario_file)

    # LOAD configuration here

    ship_list = []
    for i, pose in enumerate(pose_list):
        # ship = Ship(
        #     pose=pose,
        #     waypoints=waypoint_list[i],
        #     speed_plan=speed_plan_list[i],
        #     mmsi=i + 1,  # f'Ship{i+1}',

        # )
        ship_list.append(ship)
    return ship_list, waypoint_list, speed_plan_list


def save_scenario(pose_list, waypoint_list, speed_plan_list, savefile):
    """Saves the the scenario defined by the list of configured ships, to a json file as a dict at savefile
    dict keys:
        pose_list[i]: pose [x,y,u,psi] for ship i
        waypoint_list[i]: waypoints for ship i
        speed_plan_list[i]: speed_plan for ship i
    """

    data = {"poses": pose_list, "waypoints": waypoint_list, "speed_plans": speed_plan_list}
    with open(savefile, "w") as file:
        json.dump(data, file, indent=2)
