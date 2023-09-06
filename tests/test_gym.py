import colav_simulator.common.paths as dp
import gymnasium as gym
import numpy as np
from colav_simulator.gym.environment import COLAVEnvironment
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == "__main__":
    print(*gym.envs.registry.keys())

    config_file = dp.scenarios / "rl_scenario.yaml"

    env = gym.make(id="COLAVEnvironment-v0", scenario_config_file=config_file)
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=10_000, progress_bar=True)
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    converged = False
    env.reset()
    for i in range(1000):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))

        env.render()

        if terminated:
            print("terminated")
            break

    print("done")
