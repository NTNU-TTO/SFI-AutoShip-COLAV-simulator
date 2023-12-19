import colav_simulator.common.paths as dp
from colav_simulator.behavior_generator import BehaviorGenerationMethod, BehaviorGenerator
from colav_simulator.scenario_management import Config, ScenarioGenerator

if __name__ == "__main__":
    scenario_config = Config()
    scenario_config.behavior_generator.ownship_method = BehaviorGenerationMethod.ConstantSpeedAndCourse
    scenario_config.behavior_generator.target_ship_method = BehaviorGenerationMethod.RRTStar

    scenario_generator = ScenarioGenerator(config=scenario_config)
    scenario_generator.seed(6)  # seed = 6 crossing, seed = 15 head-on

    scenario_episode_list, scenario_enc = scenario_generator.generate(
        config_file=dp.scenarios / "boknafjorden_generation_test.yaml", new_load_of_map_data=False, show_plots=True
    )

    print("done")
