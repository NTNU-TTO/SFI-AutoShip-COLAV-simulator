import colav_simulator.common.paths as dp
from colav_simulator.scenario_management import ScenarioGenerator

if __name__ == "__main__":

    scenario_generator = ScenarioGenerator()

    scenario_episode_list, scenario_enc = scenario_generator.generate(config_file=dp.scenarios / "head_on.yaml")

    print("done")
