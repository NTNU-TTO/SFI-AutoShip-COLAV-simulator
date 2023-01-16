import colav_simulator.common.paths as dp
from colav_simulator.scenario_management import ScenarioGenerator

if __name__ == "__main__":

    scenario_generator = ScenarioGenerator()

    ship_list, scenario_config = scenario_generator.generate(scenario_config_file=dp.scenarios / "head_on.yaml")

    print("done")
