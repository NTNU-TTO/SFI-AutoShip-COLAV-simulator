"""
    environment.py

    Summary:
        This file wraps the colav-simulator for use with OpenAI Gymnasium.

    Author: Trym Tengesdal
"""

import pathlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TypeVar

import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.gym.action as actions
import colav_simulator.gym.observation as observations
import colav_simulator.scenario_management as sm
import colav_simulator.simulator as simulator
import gymnasium as gym
import numpy as np
import seacharts.enc as senc
from colav_simulator.core.ship import Ship
from gymnasium.utils import seeding

Action = TypeVar("Action")
Observation = TypeVar("Observation")


@dataclass
class Config:
    observation_type: observations.ObservationType = observations.LidarLikeObservation()
    action_type: actions.ActionType = actions.ContinuousAutopilotReferenceAction()

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(self, config_dict: dict):
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)

        config.observation_type = observations.observation_factory(self, config_dict["observation_type"])
        config.action_type = actions.action_factory(self, config_dict["action_type"])
        return config

    def to_dict(self) -> dict:
        config_dict = {}
        config_dict["observation_type"] = self.observation_type.name
        config_dict["action_type"] = self.action_type.name
        return config_dict


class BaseEnvironment(gym.Env):
    """
    A generic environment for ship collision-free planning tasks.

    The environment is centered on the own-ship (single-agent), and consists of a maritime scenario with possibly multiple other vessels and grounding hazards from ENC data.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        simulator_config: Optional[simulator.Config] = None,
        scenario_generator_config: Optional[sm.Config] = None,
        scenario_config: Optional[sm.ScenarioConfig] = None,
        render_mode: Optional[str] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        super().__init__()

        # Configuration
        self.simulator: simulator.Simulator = simulator.Simulator(config=simulator_config)
        self.scenario_generator: sm.ScenarioGenerator = sm.ScenarioGenerator(config=scenario_generator_config)
        self.scenario_config: Optional[sm.ScenarioConfig] = scenario_config if scenario_config else None

        # Scene
        self.ownship = None
        self.obstacles = None

        # Default spaces, will be set in derived classes
        self.action_type = None
        self.observation_type = None
        self._perception_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 1), dtype=np.float32)
        self._navigation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 1), dtype=np.float32)
        self._observation_space = gym.spaces.Dict({"perception": self._perception_space, "navigation": self._navigation_space})
        self._action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # Running
        self.steps = 0  # Actions performed

        self._viewer2d = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.enable_auto_render = False

        self.verbose = verbose

        self.reset()

    @classmethod
    def default_config(cls) -> Config:
        """Returns the default config for the environment.

        Can be overloaded in environment implementations, or by calling configure().

        Returns:
            dict: A configuration dict
        """
        return {
            "observation": {"type": "Kinematics"},
            "action": {"type": "DiscreteMetaAction"},
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "screen_width": 600,  # [px]
            "screen_height": 150,  # [px]
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "real_time_rendering": False,
        }

    @property
    def action_space(self) -> gym.spaces.Box:
        """Array defining the shape and bounds of the agent's action."""
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """Array defining the shape and bounds of the agent's observations."""
        return self._observation_space

    def close(self):
        """Closes the environment. To be called after usage."""
        if self._viewer2d is not None:
            self._viewer2d.close()

    def _reward(self, action: Action) -> float:
        """Returns the reward of the state-action transition.

        Args:
            action (Action): Action vector applied by the agent

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            float: Reward of the state-action transition
        """
        raise NotImplementedError

    def _is_terminated(self) -> bool:
        """Check whether the current state is a terminal state.

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            bool: Whether the current state is a terminal state
        """
        raise NotImplementedError

    def _is_truncated(self) -> bool:
        """Check whether the current state is a truncated state (time limit reached).

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            bool: Whether the current state is a truncated state
        """
        raise NotImplementedError

    def _generate(self, sconfig: Optional[sm.ScenarioConfig] = None) -> None:
        """Generates new scenarios from the configuration."""
        raise NotImplementedError

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """Returns a dictionary of additional information as defined by the environment.

        Args:
            obs (Observation): Observation vector from the environment
            action (Optional[Action]): Action vector applied by the agent. Defaults to None.

        Returns:
            dict: Dictionary of additional information
        """
        info = {
            "speed": self.ownship.speed,
            "collision": self.ownship_collision,
            "action": action,
        }
        info["rewards"] = self._rewards(action)
        return info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[observations.Observation, dict]:
        """Reset the environment to a new scenario.

        Args:
            seed (Optional[int]): Seed for the random number generator. Defaults to None.
            options (Optional[dict]): Options for the environment. Defaults to None.

        Returns:
            Tuple[observations.Observation, dict]: Initial observation and additional information
        """
        super().reset(seed=seed, options=options)
        if options and "config" in options:
            self.configure(options["config"])
        self.update_metadata()
        self.define_spaces()  # First, to set the controlled ship class depending on action space
        self.time = self.steps = 0

        self.define_spaces()  # Second, to link the obs and actions to the own-ship once the scene is created
        obs = self.observation_type.observe()  # normalized observation
        info = self._info(obs, action=self.action_space.sample())
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Perform an action in the environment and return the new observation, the reward, whether the task is terminated, and additional information.

        Args:
            action (Action): Action vector applied by the agent

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: New observation, reward, whether the task is terminated, whether the state is truncated, and additional information.
        """
        sim_data_dict = {}
        true_do_states = []

        # map action from [-1, 1] to colav system/autopilot references ranges
        sim_data_dict = self.simulator.step(action)

        if self.render_mode == "human":
            self.render()

        obs = self.observation_type.observe()  # normalized observation
        reward = self._reward(action)  # normalized reward
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Renders the environment in 2D."""
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        self.viewer.display()

        if self._time % 10.0 < 0.0001:
            self._visualizer.update_live_plot(self._time, self._enc, self._ship_list, self._most_recent_sensor_measurements[0])

        if self.render_mode == "rgb_array":
            image = self.viewer.get_image()
            return image

    @property
    def enc(self) -> senc.Enc:
        """The ENC used for the environment."""
        return self.simulator.enc

    @property
    def dynamic_obstacles(self) -> list:
        """The dynamic obstacles in the environment."""
        return self.simulator.ship_list[1:]

    @property
    def ownship(self) -> simulator.Ship:
        """The ownship in the environment."""
        return self.simulator.ship_list[0]
