"""
    behavior_generator.py

    Summary:
        Contains a class for generating random ship behaviors, i.e. trajectories.

    Author: Trym Tengesdal
"""

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import colav_simulator.behavior_generator as tg
import colav_simulator.common.config_parsing as cp
import colav_simulator.common.file_utils as file_utils
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp
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
    RapidlyExploringRandomTree = 3  # Use RRT to generate ship trajectories, with constant speed


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

    def setup(self, enc: senc.ENC, safe_sea_cdt: list) -> None:
        """Setup the environment for the behavior generator by e.g. transferring ENC data to the

        Args:
            enc (senc.ENC): Electronic navigational chart.
            safe_sea_cdt (list): List of polygons forming the safe sea Constrained Delaunay Triangulation.
        """
        self._enc = copy.deepcopy(enc)
        self._safe_sea_cdt = safe_sea_cdt

    def generate(self, rng: np.random.Generator, ship_obj: ship.Ship, random_method: Optional[bool] = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Generate a random ship behavior in the form of waypoints + speed plan that the ship will follow.

        Args:
            xs_start: Start positions of the ships.
            random_method: If True, use a random method for generating the ship behavior. If False, use the method specified in the config.
        """
        waypoints = np.zeros((0, 2))
        speed_plan = np.zeros((0, 1))
        trajectory = None

        return waypoints, speed_plan, trajectory
