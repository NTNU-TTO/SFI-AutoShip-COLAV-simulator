"""
    Test module for the visualizer result plotting functionality.
"""

import pickle
from pathlib import Path

import colav_simulator.common.paths as dp
import colav_simulator.simulator as cssim


def test_visualize_results() -> None:

    csconfig = cssim.Config.from_file(dp.config / "simulator.yaml")
    csconfig.visualizer.matplotlib_backend = "TkAgg"
    csconfig.visualizer.show_results = True
    csconfig.visualizer.show_trajectory_tracking_results = True
    csconfig.visualizer.show_target_tracking_results = True
    simulator = cssim.Simulator(config=csconfig)
    pickle_file_path = Path("simdata.pkl")
    [enc, sim_data, sim_times, ship_list] = pickle.load(pickle_file_path.open("rb"))
    simulator.visualizer.visualize_results(
        enc=enc,
        ship_list=ship_list,
        sim_data=sim_data,
        sim_times=sim_times,
        save_file_path="testres",
        pickle_input_data_for_debugging=False,
    )
    print("done")


if __name__ == "__main__":
    test_visualize_results()
