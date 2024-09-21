"""
    Test module for the Simulator class, shows how the output data is structured.

    Shows how to use the simulator with a colav system.
"""

import colav_simulator.common.paths as dp
import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.scenario_generator as sg
import colav_simulator.simulator as sim


def test_simulator_data_output() -> None:
    scenario_name = "rlmpc_scenario_ms_channel"
    scenario_generator = sg.ScenarioGenerator()
    scenario_data = scenario_generator.generate(
        config_file=dp.scenarios / (scenario_name + ".yaml"),
        new_load_of_map_data=True,
        save_scenario=True,
        save_scenario_folder=dp.scenarios / "saved" / scenario_name,
        show_plots=False,
        episode_idx_save_offset=0,
        n_episodes=1,
        delete_existing_files=True,
    )
    sbmpc_obj = ci.SBMPCWrapper()
    simulator = sim.Simulator()
    simulator.toggle_liveplot_visibility(True)
    output = simulator.run([scenario_data], colav_systems=[(0, sbmpc_obj)])

    print("Episode 1 vessel data container length:")
    print(len(output[0]["episode_simdata_list"][0]["vessel_data"]))
    print("Episode 1 sim dataframe size:")
    print(output[0]["episode_simdata_list"][0]["sim_data"].size)
    print("Episode 1 ship infos:")
    print(output[0]["episode_simdata_list"][0]["ship_info"])
    print("Scenario ENC: ")
    print(output[0]["enc"])


if __name__ == "__main__":
    test_simulator_data_output()
