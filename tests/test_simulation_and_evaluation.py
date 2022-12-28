from colav_evaluation_tools.evaluation import Evaluation
from colav_simulator.simulator import Simulator

if __name__ == "__main__":

    simulator = Simulator()

    sim_data_list, ais_data_list = simulator.run()
