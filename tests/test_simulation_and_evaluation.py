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

        evaluator.print_vessel_scores(vessel_id=15)
        evaluator.plot_trajectories_and_scores(15, [15, 17])

    print("done")
