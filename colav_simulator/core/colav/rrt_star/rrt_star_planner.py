"""
    rrt_star_planner.py

    Summary:
        A Rapidly-exploring Random Tree (RRT) based COLAV planning algorithm.

    Author: Trym Tengesdal
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from colav_simulator.core.colav.colav_interface import ICOLAV
from seacharts.enc import ENC


class RRTStar(ICOLAV):
    def __init__(self) -> None:
        self._lol = 0

    def plan(self, t: float, ownship_state: np.ndarray, do_list: list, enc: Optional[ENC] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
