import colav_simulator.common.paths as dp
import gymnasium as gym
import numpy as np
from colav_simulator.gym.environment import COLAVEnvironment
from moviepy.editor import VideoFileClip

if __name__ == "__main__":
    config_file = dp.scenarios / "rl_scenario.yaml"

    env_id = "COLAVEnvironment-v0"
    env_config = {"scenario_config_file": config_file, "render_mode": "rgb_array", "render_step_interval": 5, "test_mode": True}
    env = gym.make(id=env_id, **env_config)
    record = True
    if record:
        video_path = dp.animation_output / "demo.gif"
        env = gym.wrappers.RecordVideo(env, video_path.as_posix(), episode_trigger=lambda x: x == 0)

    env.reset(seed=1)
    for i in range(300):
        obs, reward, terminated, truncated, info = env.step(np.array([-0.2, 0.0]))

        env.render()

        if terminated or truncated:
            env.reset()
    env.close()

    # store as gif
    if record:
        vid = VideoFileClip(video_path.as_posix())
        vid.write_gif(video_path.as_posix().replace("mp4", "gif"))
    print("done")
