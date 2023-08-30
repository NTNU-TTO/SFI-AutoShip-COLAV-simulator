"""
    environment.py

    Summary:
        This file wraps the colav-simulator for use with OpenAI Gymnasium.

    Author: Trym Tengesdal
"""

import pathlib
from typing import Dict, Optional, Text, Tuple

import colav_simulator.scenario_management as scenario_management
import colav_simulator.simulator as simulator
import gymnasium as gym
from gymnasium.utils import seeding


class BaseEnvironment(gym.Env):

    """
    A generic environment for ship collision-free planning tasks.

    The environment is centered on the own-ship (single-agent), and consists of a maritime scenario with possibly multiple other vessels and grounding hazards from ENC data.
    """

    observation_type: ObservationType
    action_type: ActionType
    metadata = {"render.modes": ["human"]}

    def __init__(self, config: Optional[scenario_management.ScenarioConfig] = None, render_mode: Optional[str] = None) -> None:
        super().__init__()

        # Configuration
        self.config = self.default_config()
        self.configure(config)

        # Scene
        self.road = None
        self.controlled_vehicles = []

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False

        # Rendering
        self.viewer = None
        self._record_video_wrapper = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.enable_auto_render = False

        self.reset()

    # @classmethod
    # def default_config(cls) -> dict:
    #     """
    #     Default environment configuration.

    #     Can be overloaded in environment implementations, or by calling configure().
    #     :return: a configuration dict
    #     """
    #     return {
    #         "observation": {"type": "Kinematics"},
    #         "action": {"type": "DiscreteMetaAction"},
    #         "simulation_frequency": 15,  # [Hz]
    #         "policy_frequency": 1,  # [Hz]
    #         "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    #         "screen_width": 600,  # [px]
    #         "screen_height": 150,  # [px]
    #         "centering_position": [0.3, 0.5],
    #         "scaling": 5.5,
    #         "show_trajectories": False,
    #         "render_agent": True,
    #         "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
    #         "manual_control": False,
    #         "real_time_rendering": False,
    #     }

    # def configure(self, config: dict) -> None:
    #     if config:
    #         self.config.update(config)

    @property
    def action_space(self) -> gym.spaces.Box:
        """Array defining the shape and bounds of the agent's action."""
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Array defining the shape and bounds of the agent's observations."""
        return self._observation_space

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

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

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """Returns a dictionary of additional information as defined by the environment.

        Args:
            obs (Observation): Observation vector from the environment
            action (Optional[Action], optional): Action vector applied by the agent. Defaults to None.

        Returns:
            dict: Dictionary of additional information
        """
        info = {
            "speed": self.ownship.speed,
            "crashed": self.ownship.crashed,
            "action": action,
        }
        try:
            info["rewards"] = self._rewards(action)
        except NotImplementedError:
            pass
        return info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Observation, dict]:
        """Reset the environment to a new scenario.

        Args:
            seed (Optional[int], optional): Seed for the random number generator. Defaults to None.
            options (Optional[dict], optional): Options for the environment. Defaults to None.

        Returns:
            Tuple[Observation, dict]: Initial observation and additional information
        """
        super().reset(seed=seed, options=options)
        if options and "config" in options:
            self.configure(options["config"])
        self.update_metadata()
        self.define_spaces()  # First, to set the controlled ship class depending on action space
        self.time = self.steps = 0
        self.done = False

        # Reset scenario

        self.define_spaces()  # Second, to link the obs and actions to the own-ship once the scene is created
        obs = self.observation_type.observe()
        info = self._info(obs, action=self.action_space.sample())
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """Perform an action in the environment and return the new observation, the reward, whether the task is terminated, and additional information.


        Args:
            action (Action): Action vector applied by the agent

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            Tuple[Observation, float, bool, bool, dict]: New observation, reward, whether the task is terminated, whether the state is truncated, and additional information.
        """
        if self.road is None or self.ownship is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # def render(self) -> None:
    #     """Renders the environment in 2D."""
    #     if self.render_mode is None:
    #         assert self.spec is not None
    #         gym.logger.warn(
    #             "You are calling render method without specifying any render mode. "
    #             "You can specify the render_mode at initialization, "
    #             f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
    #         )
    #         return
    #     if self.viewer is None:
    #         self.viewer = EnvViewer(self)

    #     self.enable_auto_render = True

    #     self.viewer.display()

    #     if not self.viewer.offscreen:
    #         self.viewer.handle_events()
    #     if self.render_mode == "rgb_array":
    #         image = self.viewer.get_image()
    #         return image
