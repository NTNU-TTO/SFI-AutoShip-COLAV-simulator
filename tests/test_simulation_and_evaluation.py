from colav_evaluation_tool.evaluator import Evaluator
from colav_simulator.simulator import Simulator

if __name__ == "__main__":

    simulator = Simulator()

    output = simulator.run()
    vessels_data_list = output["vessels_data_list"]

    evaluator = Evaluator()

    for vessels in vessels_data_list:
        evaluator.set_vessel_data(vessels)
        evaluator.evaluate()

        evaluator.print_vessel_scores(vessel_id=0)
        # evaluator.plot_trajectories_and_scores(0, [0, 1])
        # evaluator.plot_scores(vessel_ids=[0, 1])
        # evaluator.plot_maneuver_detection_information(vessel_id=0)

    print("done")
