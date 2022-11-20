"""
    animation.py

    Summary:
        Contains functionality for animating the ship trajectories.

    Author: Trym Tengesdal, Magne Aune, Melih Akdag, Joachim Miller
"""
from pathlib import Path
from typing import Optional

import colav_simulator.common.map_functions as mapf
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from seacharts.enc import ENC


def visualize(
    enc: ENC,
    ship_list: list,
    data,
    times: np.ndarray,
    show_waypoints: Optional[bool] = True,
    show_animation: Optional[bool] = True,
    save_animation: Optional[bool] = True,
    save_path: Optional[Path] = None,
) -> None:
    """Visualize the ship trajectories.

    Args:
        enc (ENC): Electronic Navigational Chart object.
        ship_list
        data (DataFrame): Pandas DataFrame containing the ship simulation data.
        times (List/np.ndarray): List/array of times to consider.
        show_waypoints (Optional[bool]): _description_. Defaults to True.
        show_animation (Optional[bool]): _description_. Defaults to True.
        save_animation (Optional[bool]): Boolean flag for saving the animation. Defaults to True.
        save_path (Optional[pathlib.Path]): Path to file where the animation is saved. Defaults to None.
    """

    fig_map, ax_map = plt.subplots(figsize=(12, 10))  # facecolor=(0.8, 0.8, 0.8))
    ax_map.margins(x=0.0, y=0.0)

    mapf.plot_background(fig_map, enc, show=True)

    # make the position(circle) and speed(line) visualizing
    vessels = []
    lines = []

    n_ships = len(data.columns)

    # data: n_ships x 1 dataframe, where each ship entry contains n_samples x 1 dictionaries of pose, waypoints, speed plans etc..
    for i in range(n_ships):
        if i == 0:
            c = "b"
            vessels.append(plt.fill([], [], c, label="Own ship")[0])
            lines.append(plt.plot([], [], c + "-")[0])
        elif i == 1:
            c = "r"
            vessels.append(plt.fill([], [], c, label="Target ship")[0])
            lines.append(plt.plot([], [], c + "-")[0])
        else:
            c = "k"
            vessels.append(plt.fill([], [], "ko")[0])
            lines.append(plt.plot([], [], "k-")[0])
        if show_waypoints:
            c = "m"
            waypoints = data[f"Ship{i}"][0]["waypoints"]
            _, n_wps = waypoints.shape
            for w in range(n_wps):
                ax_map.scatter(waypoints[1, w], waypoints[0, w], color=c, s=15, alpha=0.3)
                if w < n_wps - 1:
                    ax_map.plot(
                        [waypoints[1, w], waypoints[1, w + 1]],
                        [waypoints[0, w], waypoints[0, w + 1]],
                        "--" + c,
                        alpha=0.4,
                    )

    artists = vessels  # + trajectories

    def init():
        # ax_map.set_xlim(x_lim[0], x_lim[1])
        # ax_map.set_ylim(y_lim[0], y_lim[1])
        return artists

    def update(i):
        for j in range(n_ships):
            x_i = data[f"Ship{j}"][i]["pose"][0]
            y_i = data[f"Ship{j}"][i]["pose"][1]
            U_i = data[f"Ship{j}"][i]["pose"][2]
            chi_i = data[f"Ship{j}"][i]["pose"][3]
            length = ship_list[j].length
            width = ship_list[j].width

            ship_poly = mapf.create_ship_polygon(x_i, y_i, chi_i, length, width, 1.0)
            y_ship, x_ship = ship_poly.exterior.xy

            # vessels[j].set_xy()
            vessels[j].set_xy(np.array([y_ship, x_ship]).T)

        artists = vessels
        return artists

    plt.legend(loc="upper right")

    anim = animation.FuncAnimation(
        fig_map, func=update, init_func=init, frames=len(times) - 1, repeat=False, interval=1000, blit=True
    )

    if show_animation:
        plt.show()

    if save_animation:
        anim.save(save_path, writer="ffmpeg", progress_callback=lambda i, n: print(f"Saving frame {i} of {n}"))
