"""Test module for gym.py

    Shows how to use the gym environment, and how to save a video + gif of the simulation.
"""

from pathlib import Path

import colav_simulator.common.paths as dp
import gymnasium as gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from colav_simulator.gym.environment import COLAVEnvironment
from colav_simulator.scenario_generator import ScenarioGenerator
from matplotlib import animation

# Depending on your OS, you might need to change these paths
plt.rcParams["animation.convert_path"] = "/usr/bin/convert"
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"


def save_frames_as_gif(frame_list: list, filename: Path) -> None:
    # Mess with this to change frame size
    fig = plt.figure(figsize=(frame_list[0].shape[1] / 72.0, frame_list[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frame_list[0], aspect="auto")
    plt.axis("off")

    def init():
        patch.set_data(frame_list[0])
        return (patch,)

    def animate(i):
        patch.set_data(frame_list[i])
        return (patch,)

    anim = animation.FuncAnimation(
        fig=fig, func=animate, init_func=init, blit=True, frames=len(frame_list), interval=50, repeat=True
    )
    anim.save(
        filename=filename.as_posix(),
        writer=animation.PillowWriter(fps=20),
        progress_callback=lambda i, n: print(f"Saving frame {i} of {n}"),
    )


if __name__ == "__main__":
    config_file = dp.scenarios / "rl_scenario_smaller.yaml"

    # scenario_generator = ScenarioGenerator(seed=0)
    # scenario_data = scenario_generator.generate(
    #     config_file=config_file,
    #     new_load_of_map_data=True,
    #     save_scenario=True,
    #     save_scenario_folder=dp.scenarios / "test_data" / "rl_scenario_smaller",
    #     show_plots=True,
    #     episode_idx_save_offset=0,
    #     delete_existing_files=True,
    # )

    env_id = "COLAVEnvironment-v0"
    env_config = {
        "scenario_config": config_file,
        "reload_map": False,
        "render_mode": "rgb_array",
        "render_update_rate": 1.0,
        "test_mode": False,
        "seed": 1,
    }
    env = gym.make(id=env_id, **env_config)

    record = False
    if record:
        video_path = dp.animation_output / "demo.mp4"
        env = gym.wrappers.RecordVideo(env, video_path.as_posix(), episode_trigger=lambda x: x == 0)

    env.reset(seed=1)
    frames = []
    for i in range(100):
        obs, reward, terminated, truncated, info = env.step(np.array([-0.2, 0.0]))

        frames.append(env.render())

        if terminated or truncated:
            env.reset()

    env.close()

    save_gif = True
    if save_gif:
        save_frames_as_gif(frames, dp.animation_output / "demo2.gif")

    print("done")
