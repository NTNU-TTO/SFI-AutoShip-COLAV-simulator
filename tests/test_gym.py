import colav_simulator.common.paths as dp
import gymnasium as gym
import numpy as np
from colav_simulator.gym.environment import COLAVEnvironment

if __name__ == "__main__":
    config_file = dp.scenarios / "rl_scenario.yaml"

    env_id = "COLAVEnvironment-v0"
    env_config = {"scenario_config_file": config_file, "test_mode": True}
    env = gym.make(id=env_id, **env_config)
    record = True
    if record:
        video_path = dp.videos / "demo.mp4"
        env = gym.wrappers.RecordVideo(env, video_path.as_posix(), episode_trigger=lambda x: x == 1)

    env.reset()
    for i in range(2000):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, -1.0]))

        env.render()

        if terminated or truncated:
            env.reset()
    env.close()
    print("done")
