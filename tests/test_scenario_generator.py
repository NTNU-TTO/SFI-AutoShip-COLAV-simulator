import colav_simulator.common.paths as dp
from colav_simulator.scenario_generator import ScenarioGenerator

if __name__ == "__main__":
    scenario_generator = ScenarioGenerator(seed=0)

    scenario_name = "rlmpc_scenario_ms_channel"

    # Generate scenario and save all episode .yaml files to folder
    scenario_data_list = scenario_generator.generate(
        config_file=dp.scenarios / (scenario_name + ".yaml"),
        new_load_of_map_data=True,
        save_scenario=True,
        save_scenario_folder=dp.scenarios / "saved" / scenario_name,
        show_plots=True,
        episode_idx_save_offset=0,
        n_episodes=50,
        delete_existing_files=True,
    )

    # We can then load the scenario data from the saved files with
    scenario_data = scenario_generator.load_scenario_from_folders(
        folder=dp.scenarios / "saved" / scenario_name,
        scenario_name=scenario_name,
        reload_map=False,
        max_number_of_episodes=1000,
        shuffle_episodes=False,
        show=True,
    )

    print("done")
