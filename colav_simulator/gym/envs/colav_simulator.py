"""
    colav_simulator.py

    Summary:
        This file contains the real world colav-simulator environment wrapped for use with Gymnasium.

    Author: Trym Tengesdal
"""
from typing import Optional, Tuple, Union

import colav_simulator.scenario_management as sm
import colav_simulator.simulator as csim
from colav_simulator.gym.environment import BaseEnvironment


class COLAVSimulatorEnvironment(BaseEnvironment):
    def __init__(self, config: Optional[sm.ScenarioConfig] = None, **kwargs) -> None:

        super().__init__(config, **kwargs)


    def _generate(self):