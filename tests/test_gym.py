import colav_simulator.common.paths as dp
import gymnasium as gym
import numpy as np
import stable_baselines3.common.utils as sb3_utils
import stable_baselines3.common.vec_env as sb3_vec_env
from colav_simulator.gym.environment import COLAVEnvironment
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy


def make_multiprocess_environment(env_id: str, rank: int, seed: int = 0, **kwargs):
    """Creates a multiprocess environment for stable_baselines3.

    Args:
        env_id (str): ID of the environment to create.
        rank (int): Rank of the process.
        seed (int, optional): Seed for the environment random number generator. Defaults to 0.
    """

    def _init():
        env_ = gym.make(env_id, **kwargs)
        env_.seed(seed + rank)
        env.action_space.seed(seed + rank)
        return env_

    sb3_utils.set_random_seed(seed)
    return _init


if __name__ == "__main__":
    print(*gym.envs.registry.keys())

    config_file = dp.scenarios / "rl_scenario.yaml"

    env_id = "COLAVEnvironment-v0"
    env_config = {"scenario_config_file": config_file, "test_mode": True}
    env = gym.make(id=env_id, **env_config)
    model_name = "PPO"
    num_cpu = 2
    if model_name == "DDPG" or model_name == "TD3" or model_name == "SAC":
        vec_env = sb3_vec_env.DummyVecEnv([lambda: gym.make(id=env_id, **env_config)])
    else:
        vec_env = sb3_vec_env.SubprocVecEnv([make_multiprocess_environment(env_id, i, **env_config) for i in range(num_cpu)])

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=10_000, progress_bar=True)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    converged = False
    env.reset()
    for i in range(1000):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))

        env.render()

    print("done")
