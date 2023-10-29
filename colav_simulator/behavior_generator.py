"""
    behavior_generator.py

    Summary:
        Contains a class for generating random ship behaviors/waypoints+speed plans/trajectories.

    Author: Trym Tengesdal
"""

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional, Tuple

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.core.ship as ship
import colav_simulator.core.stochasticity as stoch
import numpy as np
import seacharts.enc as senc

np.set_printoptions(suppress=True, formatter={"float_kind": "{:.2f}".format})


class BehaviorGenerationMethod(Enum):
    """Enum for the different possible methods for generating ship behaviors/waypoints."""

    ConstantSpeedAndCourse = 0  # Constant ship speed and course
    ConstantSpeedRandomWaypoints = 1  # Constant ship speed, uniform distribution to generate ship waypoints
    RandomWaypoints = 2  # Varying ship speed, uniform distribution to generate ship waypoints
    RapidlyExploringRandomTree = 3  # Use PQRRT*/RRT to generate ship trajectories/waypoints, with constant speed


@dataclass
class Config:
    """Configuration class for the behavior generator."""

    method: BehaviorGenerationMethod = BehaviorGenerationMethod.ConstantSpeedAndCourse
    n_wps_range: list = field(default_factory=lambda: [2, 4])  # Range of number of waypoints to be generated
    speed_plan_variation_range: list = field(default_factory=lambda: [-1.0, 1.0])  # Determines maximal +- change in speed plan from one segment to the next
    waypoint_dist_range: list = field(default_factory=lambda: [200.0, 1000.0])  # Range of [min, max] change in distance between randomly created waypoints
    waypoint_ang_range: list = field(default_factory=lambda: [-45.0, 45.0])  # Range of [min, max] change in angle between randomly created waypoints

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config()
        config.method = BehaviorGenerationMethod(config_dict["method"])
        return config

    def to_dict(self):
        output = asdict(self)
        output["method"] = self.method.name
        return output


class BehaviorGenerator:
    """Class for generating random ship behaviors, using one or multiple methods."""

    def __init__(self, config: Config) -> None:
        self._config: Config = config
        self._enc: senc.ENC = None
        self._safe_sea_cdt: list = None
        self._rrt_list: list = None
        self._pqrrt_list: list = None

    def setup(self, enc: senc.ENC, initial_csog_states: list) -> None:
        """Setup the environment for the behavior generator by e.g. transferring ENC data to the

        Args:
            enc (senc.ENC): Electronic navigational chart.
            initial_csog_states (list): List of initial CSOG states for the ships.
        """
        self._enc = copy.deepcopy(enc)
        self._safe_sea_cdt = mapf.create_safe_sea_triangulation(enc=self._enc, show_plots=False)

        if self._config.method == BehaviorGenerationMethod.RapidlyExploringRandomTree:
            for csog_state in initial_csog_states:
                self._rrt_list.append(rrt_lib.RRT(self._enc, csog_state))
                self._pqrrt_list.append(rrt_lib.PQRRTStar(self._enc, csog_state))

    def generate(
        self, rng: np.random.Generator, ship_list: list, ship_config_list: list, randomize_method: Optional[bool] = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Generates ship behaviors in the form of waypoints + speed plan that the ships will follow, for ships that have not been configured fully yet.

        Args:
            rng (np.random.Generator): Random number generator.
            ship_list (list): List of ships to be considered in simulation.
            ship_config_list (list): List of ship configurations.
            randomize_method: If True, randomize the method for generating the ship behavior. If False, use the method specified in the config.

        Returns:
            - Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: Tuple containing the waypoints, speed plan and possibly trajectory for the ship.
        """
        waypoints = np.zeros((0, 2))
        speed_plan = np.zeros((0, 1))
        trajectory = None

        output = {}
        for ship_cfg_idx, ship_config in enumerate(ship_config_list):
            if ship_config.random_generated:
                continue

            if ship_config.waypoints is None and ship_obj.trajectory.size < 1:
                waypoints = self.generate_random_waypoints(ship_obj.csog_state[0], ship_obj.csog_state[1], ship_obj.csog_state[3], ship_obj.draft)
                speed_plan = self.generate_random_speed_plan(ship_obj.csog_state[2], U_min=ship_obj.min_speed, U_max=ship_obj.max_speed, n_wps=waypoints.shape[1])
                ship_config.waypoints = waypoints
                ship_config.speed_plan = speed_plan

            ship_obj.set_nominal_plan(ship_config.waypoints, ship_config.speed_plan)

            csog_state_list.append(ship_obj.csog_state)
            ship_list.append(ship_obj)
            ship_config_list.append(ship_config)
            cfg_ship_idx += 1

        output["ship_list"] = ship_list
        output["ship_config_list"] = ship_config_list
        output["csog_state_list"] = csog_state_list
        output["non_cfged_ship_indices"] = non_cfged_ship_indices
        output["cfg_ship_idx"] = cfg_ship_idx
        return output
