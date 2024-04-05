"""
    reporter.py

    Summary:
        Contains class for reporting data from the RL training process.

    Author: Trym Tengesdal
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tables
from colav_simulator.gym.environment import COLAVEnvironment


class Log(tables.IsDescription):
    episode = tables.Int32Col()
    timesteps = tables.Int32Col()
    duration = tables.Float32Col()
    goal_reached = tables.Int32Col()
    distance_to_grounding = tables.Float32Col()
    distance_to_collision = tables.Float32Col()
    collision = tables.Int32Col()
    grounding = tables.Int32Col()
    reward = tables.Float32Col()
    cumulative_reward = tables.Float32Col()


class Logger:
    def __init__(self, log_dir: Path, experiment_name: str) -> None:
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        self.file = tables.open_file(log_dir / (experiment_name + ".h5"), mode="w", title=experiment_name)
        env_group = self.file.create_group("/", "env", "Environment data")
        self.table = self.file.create_table(env_group, "data", Log, "Log of environment data")

    def save_and_close_hdf5(self) -> None:
        self.file.close()

    def load_hdf5(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        self.file = tables.open_file(file_path, mode="r")
        self.table = self.file.root.env.data
        logdata = self.table.read()
        self.file.close()
        return logdata

    def __call__(self, env: COLAVEnvironment) -> None:
        """Logs data from the environment in the current step to a HDF5 file.

        Args:
            env (COLAVEnvironment): The environment to log data from.
        """
        info = env.get_wrapper_attr("last_info")
        log_k = self.table.row
        log_k["episode"] = info["episode"]
        log_k["timesteps"] = info["timesteps"]
        log_k["duration"] = info["duration"]
        log_k["goal_reached"] = info["goal_reached"]
        log_k["distance_to_grounding"] = info["distance_to_grounding"]
        log_k["distance_to_collision"] = info["distance_to_collision"]
        log_k["collision"] = info["collision"]
        log_k["grounding"] = info["grounding"]
        log_k["reward"] = info["reward"]
        log_k["cumulative_reward"] = info["cumulative_reward"]
        log_k.append()
        self.table.flush()


if __name__ == "__main__":
    logger = Logger(Path("./logs"), "test")
    logger.load_hdf5(Path("home/doctor/Desktop/machine_learning/rlmpc/test.h5"))

    print("Logger loaded successfully.")
