import colav_simulator.common.paths as dp
from colav_simulator.scenario_management import ScenarioGenerator

if __name__ == "__main__":
    scenario_generator = ScenarioGenerator()
    scenario_generator.seed(12)  # seed = 5 crossing, seed = 0 head-on

    scenario_episode_list, scenario_enc = scenario_generator.generate(
        config_file=dp.scenarios / "boknafjorden_generation_test.yaml"
    )

    print("done")
