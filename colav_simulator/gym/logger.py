"""
    reporter.py

    Summary:
        Contains class for reporting data from the RL training process.

    Author: Trym Tengesdal
"""

from pathlib import Path
from typing import List, Optional, Tuple

import colav_simulator.gym.environment as csgym_env
import numpy as np
import tables


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
    max_reward = tables.Float32Col()
    min_reward = tables.Float32Col()
    mean_reward = tables.Float32Col()


class Logger:
    def __init__(self, experiment_name: str, log_dir: Optional[Path] = None) -> None:
        if log_dir is not None:
            if not log_dir.exists():
                log_dir.mkdir(parents=True)
            self.file = tables.open_file(log_dir / (experiment_name + ".h5"), mode="w", title=experiment_name)
            env_group = self.file.create_group("/", "env", "Environment data")
            self.table = self.file.create_table(env_group, "data", Log, "Log of environment data")
        self.rewards: List[float] = []
        self.max_reward: float = -np.inf
        self.min_reward: float = np.inf
        self.mean_reward: float = 0.0
        self.std_reward: float = 0.0
        self.episodes: int = 0
        self.timesteps: int = 0

    def save_and_close_hdf5(self) -> None:
        self.file.close()

    def load_hdf5(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        self.file = tables.open_file(file_path, mode="r")
        self.table = self.file.root.env.data
        logdata = self.table.read()
        self.file.close()
        return logdata

    def __call__(self, env: csgym_env.COLAVEnvironment) -> None:
        """Logs data from the environment in the current step to a HDF5 file.

        Args:
            env (COLAVEnvironment): The environment to log data from.
        """
        info = env.get_wrapper_attr("last_info")
        reward = info["reward"]
        self.rewards.append(reward)
        self.timesteps += info["timesteps"]
        self.max_reward = max(self.rewards)
        self.min_reward = min(self.rewards)
        self.mean_reward = float(np.mean(self.rewards))
        self.std_reward = float(np.std(self.rewards))

        log_k = self.table.row
        log_k["episode"] = info["episode"]
        log_k["timesteps"] = info["timesteps"]
        log_k["duration"] = info["duration"]
        log_k["goal_reached"] = info["goal_reached"]

        log_k["distance_to_grounding"] = info["distance_to_grounding"]
        log_k["distance_to_collision"] = info["distance_to_collision"]
        log_k["collision"] = info["collision"]
        log_k["grounding"] = info["grounding"]
        log_k["reward"] = reward
        log_k["max_reward"] = self.max_reward
        log_k["min_reward"] = self.min_reward
        log_k["mean_reward"] = self.mean_reward
        log_k["std_reward"] = self.std_reward
        log_k["cumulative_reward"] = sum(self.rewards)
        log_k.append()
        self.table.flush()


if __name__ == "__main__":
    logger = Logger("test")
    logger.load_hdf5(Path("/Users/trtengesdal/Desktop/machine_learning/rlmpc/test_sac_rlmpc/test_sac_rlmpc.h5"))

    print("Logger loaded successfully.")
