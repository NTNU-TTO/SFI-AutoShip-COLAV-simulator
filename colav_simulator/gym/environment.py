"""
    environment.py

    Summary:
        This file wraps the colav-simulator for use with Gymnasium.

    Author: Trym Tengesdal
"""
import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

import colav_simulator.gym.reward as rw
import colav_simulator.scenario_management as sm
import colav_simulator.simulator as cssim
import gymnasium as gym
import numpy as np
import seacharts.enc as senc
from colav_simulator.gym.action import Action, ActionType, action_factory
from colav_simulator.gym.observation import Observation, ObservationType, observation_factory


@dataclass
class Config:
    """Configuration for the environment. This excludes simulator + scenario generator config,"""

    simulator: Optional[cssim.Config] = None
    scenario_generator: Optional[sm.Config] = None
    scenario_config: Optional[sm.ScenarioConfig] = None
    scenario_config_file: Optional[pathlib.Path] = None
    rewarder: Optional[list] = None

    @classmethod
    def from_dict(self, config_dict: dict):
        cfg = Config()
        cfg.simulator = cssim.Config.from_dict(config_dict["simulator"])
        cfg.scenario_generator = sm.Config.from_dict(config_dict["scenario_generator"])

        if "scenario_config" in config_dict:
            cfg.scenario_config = sm.ScenarioConfig.from_dict(config_dict["scenario_config"])
            cfg.scenario_config_file = None

        if "scenario_config_file" in config_dict:
            cfg.scenario_config_file = pathlib.Path(config_dict["scenario_config_file"])
            cfg.scenario_config = None
        cfg.rewarder = config_dict["rewarder"]
        return cfg


class COLAVEnvironment(gym.Env):
    """
    An environment for performing ship collision-free planning tasks.

    The environment is centered on the own-ship (single-agent), and consists of a maritime scenario with possibly multiple other vessels and grounding hazards from ENC data.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    observation_type: ObservationType
    action_type: ActionType
    scenario_config: sm.ScenarioConfig
    scenario_data_tup: tuple

    def __init__(
        self,
        simulator_config: Optional[cssim.Config] = None,
        scenario_generator_config: Optional[sm.Config] = None,
        scenario_config: Optional[sm.ScenarioConfig] = None,
        scenario_config_file: Optional[pathlib.Path] = None,
        rewarder_config: Optional[rw.Config] = None,
        render_mode: Optional[str] = None,
        verbose: Optional[bool] = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.simulator: cssim.Simulator = cssim.Simulator(config=simulator_config)
        self.scenario_generator: sm.ScenarioGenerator = sm.ScenarioGenerator(config=scenario_generator_config)
        self.rewarder = rw.Rewarder(config=rewarder_config)

        self._viewer2d = self.simulator.visualizer
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.verbose = verbose

        self.reset(scenario_config=scenario_config, scenario_config_file=scenario_config_file)

    def close(self):
        """Closes the environment. To be called after usage."""
        if self._viewer2d is not None:
            self._viewer2d.close_live_plot()

    def _define_spaces(self) -> None:
        """Defines the action and observation spaces."""
        self.action_type = action_factory(self, self.scenario_config.rl_action_type)
        self.action_space = self.action_type.space()

        self.observation_type = observation_factory(self, self.scenario_config.rl_observation_type)
        self.observation_space = self.observation_type.space()

    def _is_terminated(self) -> bool:
        """Check whether the current state is a terminal state.

        Returns:
            bool: Whether the current state is a terminal state
        """
        collided = self.simulator.determine_ownship_collision()
        grounded = self.simulator.determine_ownship_grounding()
        return collided or grounded

    def _is_truncated(self) -> bool:
        """Check whether the current state is a truncated state (time limit reached).

        Returns:
            bool: Whether the current state is a truncated state
        """
        return self.simulator.t > self.simulator.t_end

    def _generate(self, sconfig: Optional[sm.ScenarioConfig] = None, config_file: Optional[pathlib.Path] = None) -> None:
        """Generates new scenarios from the configuration."""
        if sconfig is not None:
            self.scenario_data_tup = self.scenario_generator.generate(config=sconfig)
            self.scenario_config = sconfig
        elif config_file is not None:
            self.scenario_data_tup = self.scenario_generator.generate(config_file=config_file)
            self.scenario_config = self.scenario_data_tup[0][0]["config"]
        else:
            self.scenario_data_tup = self.scenario_generator.generate_configured_scenarios()[0]
            self.scenario_config = self.scenario_data_tup[0][0]["config"]

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """Returns a dictionary of additional information as defined by the environment.

        Args:
            obs (Observation): Observation vector from the environment
            action (Optional[Action]): Action vector applied by the agent. Defaults to None.

        Returns:
            dict: Dictionary of additional information
        """
        assert self.ownship is not None, "Environment not initialized!"
        info = {
            "speed": self.ownship.csog_state[2],
            "course": self.ownship.csog_state[3],
            "position": self.ownship.csog_state[0:2],
            "collision": self.simulator.determine_ownship_collision(),
            "grounding": self.simulator.determine_ownship_grounding(),
            "action": action,
            "obs": obs,
            "reward": self.rewarder(obs, action),
        }
        return info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, **kwargs) -> Tuple[Observation, dict]:
        """Reset the environment to a new scenario episode. If a scenario config or config file is provided, a new scenario is generated. Otherwise, the next episode of the current scenario is used, if any.

        Args:
            seed (Optional[int]): Seed for the random number generator. Defaults to None.
            options (Optional[dict]): Options for the environment. Defaults to None.

        Returns:
            Tuple[Observation, dict]: Initial observation and additional information
        """
        super().reset(seed=seed, options=options)
        self.steps = 0  # Actions performed
        self.scenario_generator.seed(seed=seed)
        if "scenario_config" in kwargs:
            self._generate(sconfig=kwargs["scenario_config"])
        elif "scenario_config_file" in kwargs:
            self._generate(config_file=kwargs["scenario_config_file"])

        (scenario_episode_list, scenario_enc) = self.scenario_data_tup

        episode_data = scenario_episode_list.pop(0)
        if len(scenario_episode_list) == 0:
            print("No episodes left in scenario. Re-generating the existing one.")
            self._generate(sconfig=self.scenario_config)

        self.simulator.initialize_scenario_episode(
            ship_list=episode_data["ship_list"], sconfig=episode_data["config"], enc=scenario_enc, disturbance=episode_data["disturbance"], ownship_colav_system=None
        )
        self.ownship = self.simulator.ownship

        self._define_spaces()

        obs = self.observation_type.observe()  # normalized observation
        info = self._info(obs, action=self.action_space.sample())
        if self.render_mode == "human":
            self._init_render()

        return obs, info

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """Perform an action in the environment and return the new observation, the reward, whether the task is terminated, and additional information.

        Args:
            action (Action): Action vector applied by the agent

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: New observation, reward, whether the task is terminated, whether the state is truncated, and additional information.
        """
        # map action from [-1, 1] to colav system/autopilot references ranges

        self.action_type.act(action)
        sim_data_dict = self.simulator.step()

        obs = self.observation_type.observe()  # normalized observation
        reward = self.rewarder(obs, action)  # normalized reward
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        self.steps += 1

        return obs, reward, terminated, truncated, info

    def _init_render(self) -> None:
        """Initializes the renderer."""
        self._viewer2d.init_live_plot(self.enc, self.simulator.ship_list)

    def render(self, step_interval: int = 10) -> None:
        """Renders the environment in 2D at the given step interval.

        Args:
            step_interval (int): The step interval at which to render the environment. Defaults to 10.
        """
        if self.steps % step_interval == 0:
            self._viewer2d.update_live_plot(self.simulator.t, self.enc, self.simulator.ship_list, self.simulator.recent_sensor_measurements)

    @property
    def enc(self) -> senc.ENC:
        """The ENC used for the environment."""
        return self.simulator.enc

    @property
    def dynamic_obstacles(self) -> list:
        """The dynamic obstacles in the environment."""
        return self.simulator.ship_list[1:]

    @property
    def relevant_grounding_hazards(self) -> list:
        """The nearby ownship grounding hazards in the environment."""
        return self.simulator.relevant_grounding_hazards
