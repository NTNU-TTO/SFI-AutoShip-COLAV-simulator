import colav_simulator.common.paths as dp
from colav_simulator.scenario_management import ScenarioGenerator

if __name__ == "__main__":
    scenario_generator = ScenarioGenerator()
    scenario_generator.seed(12)  # seed = 12 crossing, seed = 19 head-on

    scenario_episode_list, scenario_enc = scenario_generator.generate(
        config_file=dp.scenarios / "boknafjorden_generation_test.yaml", new_load_of_map_data=False
    )

    print("done")
