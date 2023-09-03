import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == "__main__":
    env = gym.make("COLAVSimulator-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000, progress_bar=True)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward2:{mean_reward:.2f} +/- {std_reward:.2f}")

    vec_env = model.get_env()

    obs = vec_env.reset()

    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = model.predict(observation, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()
