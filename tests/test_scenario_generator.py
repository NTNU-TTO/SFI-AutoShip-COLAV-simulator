import pathlib

from colav_simulator.scenario_generator import ScenarioGenerator

if __name__ == "__main__":

    config_file = pathlib.Path.cwd().parents[2] / "config" / "simulator.yaml"

    scenario_generator = ScenarioGenerator()

    ship_list = scenario_generator.generate()

    print("done")
