from colav_evaluation_tool.evaluator import Evaluator
from colav_simulator.simulator import Simulator

if __name__ == "__main__":
    # ScenaroGenerator.

    simulator = Simulator()

    output = simulator.run()
    episode_simdata_list = output[0]["episode_simdata_list"]
    episode = episode_simdata_list[0]
    enc = output[0]["enc"]

    evaluator = Evaluator()

    count = 0
    for episode_data in episode_simdata_list:
        vessels = episode_data["vessel_data"]
        print("Evaluating scenario " + str(count) + " with " + str(len(vessels)) + " vessels...")

        evaluator.set_enc(enc)
        evaluator.set_vessel_data(vessels)
        results = evaluator.evaluate()

        evaluator.print_vessel_scores(vessel_id=0)
        # evaluator.print_vessel_scores(vessel_id=1)
        # evaluator.plot_trajectories_and_scores(0, [0, 1])
        # evaluator.plot_scores(vessel_ids=[0, 1])
        # evaluator.plot_maneuver_detection_information(vessel_id=0)

        count += 1
    print("done")
