import pathlib

from colav_simulator.simulator import Simulator

if __name__ == "__main__":

    config_file = pathlib.Path.cwd().parents[2] / "config" / "simulator.yaml"

    simulator = Simulator()

    simulator.run()

    print("done")
