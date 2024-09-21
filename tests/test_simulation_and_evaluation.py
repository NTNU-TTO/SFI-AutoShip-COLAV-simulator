    """
    This script demonstrates how to run the COLAV simulator and evaluator together.
    """
import time

import numpy as np
from colav_evaluation_tool.evaluator import Evaluator
from colav_simulator.scenario_generator import ScenarioGenerator
from colav_simulator.simulator import Simulator


def test_simulation_and_evaluation() -> None:
    scenario_generator = ScenarioGenerator()
    simulator = Simulator()
    evaluator = Evaluator()

    scenario_list = scenario_generator.generate_configured_scenarios()
    simulator.toggle_liveplot_visibility(False)
    framework_exec_times = []
    n_runs = 1
    for i in range(n_runs):
        start_time = time.time()

        scenario_result_data_list = simulator.run(scenario_list)
        for scenario_data in scenario_result_data_list:
            episode_simdata_list = scenario_data["episode_simdata_list"]
            evaluator.set_enc(scenario_data["enc"])

            count = 0
            for episode_data in episode_simdata_list:
                vessels = episode_data["vessel_data"]
                print("Evaluating scenario " + str(count) + " with " + str(len(vessels)) + " vessels...")

                evaluator.set_vessel_data(vessels)
                results = evaluator.evaluate()

                evaluator.print_vessel_scores(vessel_id=0)
                evaluator.print_vessel_scores(vessel_id=1)
                # evaluator.plot_trajectories_and_scores(0, [0, 1])
                # evaluator.plot_scores(vessel_ids=[0, 1])
                # evaluator.plot_maneuver_detection_information(vessel_id=0)

                count += 1

        execution_time = time.time() - start_time
        print("Framework execution time in seconds: " + str(execution_time))
        framework_exec_times.append(execution_time)

    std_dev_exec_time = np.std(np.array(framework_exec_times))
    print("Average framework execution time in seconds: " + str(sum(framework_exec_times) / n_runs))
    print("Standard deviation of framework execution time in seconds: " + str(std_dev_exec_time))


if __name__ == "__main__":
    test_simulation_and_evaluation()
