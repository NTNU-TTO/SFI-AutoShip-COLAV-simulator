from colav_evaluation_tool.evaluator import Evaluator
from colav_simulator.simulator import Simulator

if __name__ == "__main__":

    simulator = Simulator()

    output = simulator.run()
    vessels_data_list = output["vessels_data_list"]
    scenario_config_list = output["scenario_config_list"]
    scenario_enc_list = output["scenario_enc_list"]

    evaluator = Evaluator()

    count = 0
    for vessels in vessels_data_list:
        print("Evaluating scenario " + str(count) + " with " + str(len(vessels)) + " vessels...")

        evaluator.set_enc(scenario_enc_list[count])
        evaluator.set_vessel_data(vessels)
        results = evaluator.evaluate()

        evaluator.print_vessel_scores(vessel_id=0)
        # evaluator.print_vessel_scores(vessel_id=1)
        # evaluator.plot_trajectories_and_scores(0, [0, 1])
        # evaluator.plot_scores(vessel_ids=[0, 1])
        # evaluator.plot_maneuver_detection_information(vessel_id=0)

        count += 1
    print("done")
