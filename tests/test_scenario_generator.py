import colav_simulator.common.paths as dp
from colav_simulator.scenario_generator import ScenarioGenerator

if __name__ == "__main__":
    scenario_generator = ScenarioGenerator(seed=0)

    scenario_name = "rlmpc_scenario_ms_channel"

    scenario_data_list = scenario_generator.generate(
        config_file=dp.scenarios / (scenario_name + ".yaml"),
        new_load_of_map_data=True,
        save_scenario=True,
        save_scenario_folder=dp.scenarios / "test" / scenario_name,
        show_plots=True,
        episode_idx_save_offset=0,
        n_episodes=50,
        delete_existing_files=True,
    )

    print("done")
