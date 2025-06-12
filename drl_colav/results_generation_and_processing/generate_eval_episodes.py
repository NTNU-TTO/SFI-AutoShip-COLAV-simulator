import time
import pathlib

import numpy as np
from colav_evaluation_tool.evaluator import Evaluator
from colav_simulator.scenario_generator import ScenarioGenerator
import colav_simulator.simulator as sim
import colav_simulator.core.colav.colav_interface as ci

def main():
    root = pathlib.Path(__file__).parents[1]
    scenario_generator = ScenarioGenerator()
    target_folder = root / "scenario/test"
    
    for f in target_folder.iterdir():
        config_file = f

        _, _ = scenario_generator.generate(
            config_file=config_file,
            save_scenario=True,
            save_scenario_folder=target_folder,
            n_episodes=1,
        )

if __name__ == "__main__":
    main()