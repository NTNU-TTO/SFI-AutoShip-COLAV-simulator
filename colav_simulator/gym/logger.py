"""
    logger.py

    Summary:
        Contains class for logging data from the COLAV environment.

    Author: Trym Tengesdal
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.plotters as plotters
import cv2
import numpy as np
import scipy.ndimage as scimg


class EpisodeData(NamedTuple):
    """Data for a single episode in the COLAV environment."""

    name: str
    episode: int
    timesteps: int
    duration: float
    distances_to_grounding: np.ndarray
    distances_to_collision: np.ndarray
    goal_reached: bool
    collision: bool
    grounding: bool
    truncated: bool
    rewards: np.ndarray
    cumulative_reward: float
    mean_reward: float
    std_reward: float
    reward_components: List[Dict[str, float]]
    frames: Optional[List[np.ndarray]]
    actions: List[np.ndarray]
    obs: List[np.ndarray]
    actor_infos: List[Dict[str, Any]]
    actor_failure: bool = False


class Logger:
    """Logs data from the COLAV environment. Supports saving and loading to/from pickle files.

    Args:
        experiment_name (str): The name of the experiment.
        log_dir (Path): The directory where the log files are saved.
        save_freq (int, optional): The frequency (in episodes) of saving the data to a pickle file. Defaults to 10.
        n_envs (int, optional): The number of environments. Defaults to 1.
        max_num_logged_episodes (int, optional): The maximum number of episodes to log before saving and resetting. Defaults to 500.

    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: Path,
        n_envs: int = 1,
        max_num_logged_episodes: int = 100,
    ) -> None:
        if not log_dir.exists():
            log_dir.mkdir(parents=True)

        self.max_num_logged_episodes: int = max_num_logged_episodes
        self.env_data: List[EpisodeData] = [
            EpisodeData(
                "",
                0,
                0,
                0.0,
                np.array([]),
                np.array([]),
                False,
                False,
                False,
                False,
                np.array([]),
                0.0,
                0.0,
                0.0,
                [],
                None,
                [],
                [],
                [],
            )
        ] * max_num_logged_episodes
        self.pos: int = 0
        self.prev_pos: int = 0
        self.experiment_name: str = experiment_name
        self.log_dir: Path = log_dir

        self.log_frame_freq: int = 4
        self.log_count: List[int] = [0 for _ in range(n_envs)]

        self.episode_name: List[str] = ["" for _ in range(n_envs)]
        self.duration: List[float] = [0.0 for _ in range(n_envs)]
        self.rewards: List[float] = [[] for _ in range(n_envs)]
        self.timesteps: List[int] = [0 for _ in range(n_envs)]
        self.prev_episode_nr: int = -1
        self.episode_nr: int = 0

        self.reward_components: List[List[Dict[str, float]]] = [[] for _ in range(n_envs)]
        self.collision: List[bool] = [False for _ in range(n_envs)]
        self.grounding: List[bool] = [False for _ in range(n_envs)]
        self.goal_reached: List[bool] = [False for _ in range(n_envs)]
        self.truncated: List[bool] = [False for _ in range(n_envs)]
        self.frames: List[List[np.ndarray]] = [[] for _ in range(n_envs)]
        self.distances_to_grounding: List[List[float]] = [[] for _ in range(n_envs)]
        self.distances_to_collision: List[List[float]] = [[] for _ in range(n_envs)]
        self.actions: List[List[np.ndarray]] = [[] for _ in range(n_envs)]
        self.obs: List[List[np.ndarray | Dict[str, np.ndarray]]] = [[] for _ in range(n_envs)]
        self.actor_infos: List[List[Dict[str, Any]]] = [[] for _ in range(n_envs)]
        self.actor_failure: List[bool] = [False for _ in range(n_envs)]

    def save_as_pickle(self, name: Optional[str] = None) -> None:
        """Saves the environment data to a pickle file.

        Args:
            name (Optional[str]): The name of the pickle file, without the .pkl extension.
        """
        if name is None:
            name = self.experiment_name + "_env_training_data"

        # Don't save if there is no data or if the episode number hasn't changed
        if self.pos == 0 or self.pos == self.prev_pos:
            return

        with open(self.log_dir / (name + ".pkl"), "ba") as f:
            pickle.dump(self.env_data[: self.pos], f)

        self.prev_pos = self.pos

    def load_from_pickle(self, name: Optional[str]) -> None:
        """Loads the environment data from a pickle file.
        Args:
            name (Optional[str]): Name of the pickle file, without the .pkl extension.
        """
        if name is None:
            name = self.experiment_name
        # Because multiple objects might be stored in the same file
        # we need to load them one by one
        self.env_data = []
        with open(self.log_dir / (name + ".pkl"), "rb") as f:
            while 1:
                try:
                    edata = pickle.load(f)
                    self.env_data.extend(edata)
                except EOFError:
                    break
        # prune equal episodes by checking the episode number
        self.env_data = list({edata.episode: edata for edata in self.env_data}.values())
        # print(f"Loaded {len(self.env_data)} episodes from {name}.pkl")

    def __call__(self, cs_env_infos: List[Dict[str, Any]]) -> None:
        """Logs data from the input env info dictionary

        Args:
            cs_env_infos (List[Dict[str, Any]]): List of environment info dictionaries for the COLAV environment(s).
        """
        for env_idx, info in enumerate(cs_env_infos):
            self._log_env_info(info, env_idx)

    def _log_env_info(self, info: Dict[str, Any], env_idx: int) -> None:
        """Logs the environment info to the logger.

        Args:
            info (Dict[str, Any]): The environment info dictionary.
            env_idx (int): The index of the environment.
        """
        self.episode_name[env_idx] = info["episode_name"]
        self.duration[env_idx] = info["duration"]
        self.timesteps[env_idx] = info["timesteps"]

        self.rewards[env_idx].append(info["reward"])
        self.collision[env_idx] = info["collision"]
        self.grounding[env_idx] = info["grounding"]
        self.goal_reached[env_idx] = info["goal_reached"]
        self.truncated[env_idx] = info["truncated"]
        self.actor_failure[env_idx] = info["actor_failure"]

        self.distances_to_collision[env_idx].append(info["distance_to_collision"])
        self.distances_to_grounding[env_idx].append(info["distance_to_grounding"])
        # self.actions[env_idx].append(info["action"])
        # self.obs[env_idx].append(info["observation"])
        self.reward_components[env_idx].append(info["reward_components"])

        # FIX logging disturbance data between steps
        # if info["disturbance"] is not None:
        #     self.disturbance_data[env_idx]

        if (
            info["render_frame"] is not None
            and info["render_frame"].size > 10
            and (self.log_count[env_idx] % self.log_frame_freq == 0)
        ):
            stored_frame = info["render_frame"].copy()

            rotate = False
            if rotate:
                stored_frame = scimg.rotate(stored_frame, np.rad2deg(info["os_course"]), reshape=False)

            npx, npy = stored_frame.shape[:2]
            center_pixel_x = int(stored_frame.shape[0] // 2)
            center_pixel_y = int(stored_frame.shape[1] // 2)
            if rotate:
                cutoff_index_below_vessel = int(0.1 * npx)  # corresponds to 100 m for a 1200 m zoom width
                cutoff_index_above_vessel = int(0.4 * npx)
                cutoff_laterally = int(0.25 * npy)
            else:
                cutoff_index_below_vessel = int(0.3 * npx)
                cutoff_index_above_vessel = int(0.3 * npx)
                cutoff_laterally = int(0.3 * npy)
            cutoff_laterally = cutoff_laterally if cutoff_laterally <= center_pixel_y else center_pixel_y
            cutoff_index_below_vessel = (
                cutoff_index_below_vessel if cutoff_index_below_vessel <= center_pixel_y else center_pixel_y
            )
            cutoff_index_above_vessel = (
                cutoff_index_above_vessel if cutoff_index_above_vessel <= center_pixel_x else center_pixel_x
            )
            cropped_img = stored_frame[
                center_pixel_x - cutoff_index_above_vessel : center_pixel_x + cutoff_index_below_vessel,
                center_pixel_y - cutoff_laterally : center_pixel_y + cutoff_laterally,
            ]
            reduced_frame = cv2.resize(cropped_img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.uint8)
            # plotters.plot_image(image=reduced_frame, title="Reduced frame for logging")
            self.frames[env_idx].append(reduced_frame)

        # Special case for an NMPC actor
        if "actor_info" in info and "soln" in info["actor_info"]:
            actor_info = info["actor_info"]
            stored_actor_info = {}
            stored_actor_info["mpc_runtime"] = actor_info["runtime"]
            stored_actor_info["mpc_cost"] = actor_info["cost_val"]
            stored_actor_info["optimal"] = actor_info["optimal"]
            # stored_actor_info["predicted_trajectory"] = actor_info["trajectory"]
            stored_actor_info["n_iterations"] = actor_info["n_iter"]
            stored_actor_info["so_constr_inf_norm"] = float(actor_info["so_constr_vals"].max())
            stored_actor_info["do_constr_inf_norm"] = float(actor_info["do_constr_vals"].max())
            stored_actor_info["dnn_input_features"] = actor_info["dnn_input_features"]
            stored_actor_info["norm_old_mpc_params"] = actor_info["norm_old_mpc_params"]
            stored_actor_info["old_mpc_params"] = actor_info["old_mpc_params"]
            stored_actor_info["new_mpc_params"] = actor_info["new_mpc_params"]
            stored_actor_info["norm_mpc_action"] = actor_info["norm_mpc_action"]
            self.actor_infos[env_idx].append(stored_actor_info)

        done = (
            self.collision[env_idx]
            or self.grounding[env_idx]
            or self.goal_reached[env_idx]
            or self.truncated[env_idx]
            or self.actor_failure[env_idx]
        )

        self.log_count[env_idx] += 1

        if done:
            self.add_episode_data(env_idx)
            self.reset_data_structures(env_idx)

    def add_episode_data(self, env_idx) -> None:
        """Adds the data from the current episode to the environment data list.

        Args:
            env_idx (int): The index of the environment.
        """
        # Don't log if there is only one timestep
        if len(self.rewards[env_idx]) == 1 or self.episode_nr == self.prev_episode_nr:
            return

        episode_data = EpisodeData(
            name=self.episode_name[env_idx],
            episode=self.episode_nr,
            timesteps=self.timesteps[env_idx],
            duration=self.duration[env_idx],
            rewards=np.array(self.rewards[env_idx], dtype=np.float32),
            cumulative_reward=np.sum(self.rewards[env_idx], dtype=np.float32),
            mean_reward=np.mean(self.rewards[env_idx], dtype=np.float32),
            std_reward=np.std(self.rewards[env_idx], dtype=np.float32),
            reward_components=self.reward_components[env_idx],
            goal_reached=self.goal_reached[env_idx],
            collision=self.collision[env_idx],
            grounding=self.grounding[env_idx],
            truncated=self.truncated[env_idx],
            distances_to_grounding=np.array(self.distances_to_grounding[env_idx], dtype=np.float32),
            distances_to_collision=np.array(self.distances_to_collision[env_idx], dtype=np.float32),
            actions=self.actions[env_idx],
            obs=self.obs[env_idx],
            frames=self.frames[env_idx],
            actor_infos=self.actor_infos[env_idx],
            actor_failure=self.actor_failure[env_idx],
        )
        self.env_data[self.pos] = episode_data
        self.prev_episode_nr = self.episode_nr
        self.pos += 1
        self.episode_nr += 1
        # mhm.print_process_memory_usage(prefix_str=f"Env {env_idx} | Episode {self.episode_nr} |")
        if self.pos >= self.max_num_logged_episodes:
            self.save_as_pickle()
            self.pos = 0
            self.prev_pos = 0

    def reset_data_structures(self, env_idx: int) -> None:
        """Resets the data structures in preparation of new episode.

        Args:
            env_idx (int): The index of the environment.
        """
        self.log_count[env_idx] = 0
        self.timesteps[env_idx] = 0
        self.duration[env_idx] = 0.0
        self.rewards[env_idx] = []
        self.reward_components[env_idx] = []
        self.goal_reached[env_idx] = False
        self.collision[env_idx] = False
        self.grounding[env_idx] = False
        self.truncated[env_idx] = False
        self.distances_to_grounding[env_idx] = []
        self.distances_to_collision[env_idx] = []
        self.actions[env_idx] = []
        self.obs[env_idx] = []
        self.frames[env_idx] = []
        self.actor_infos[env_idx] = []
        self.actor_failure[env_idx] = False


if __name__ == "__main__":
    experiment_name = "sac_rlmpc1"
    log_dir = Path.home() / "Desktop" / "machine_learning" / "rlmpc" / experiment_name
    logger = Logger(experiment_name=experiment_name, log_dir=log_dir)
    logger.load_from_pickle(f"{experiment_name}_env_training_data")

    print(f"Ep0: {logger.env_data[0]}")
