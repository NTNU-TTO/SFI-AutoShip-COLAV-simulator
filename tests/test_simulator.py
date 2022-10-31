import colav_simulator.utils.math_functions as mf
import numpy as np
from colav_simulator.ships.ship import Ship
from matplotlib import pyplot as plt

legend_size = 10  # legend size
fig_size = [25, 13]  # figure1 size in cm
dpi_value = 150  # figure dpi value

if __name__ == "__main__":

    waypoints = np.array(
        [
            [0, 50, 200, 0, -200, 0],
            [0, 50, 200, 200, 0, 0],
        ]
    )
    speed_plan = np.array([6.0, 6.0, 6.0, 6.0, 6.0, 6.0])
    state = np.array([0.0, 0.0, 0.0, 0.0])
    mmsi = "lol"
    ship = Ship(mmsi, waypoints, speed_plan, state)

    horizon = 200.0
    dt = 1.0
    n = 4  # when using kinematic model considering only pos and vel
    n_samples = round(horizon / dt)
    trajectory = np.zeros((n, n_samples))
    time = np.zeros(n_samples)
    for k in range(n_samples):
        time[k] = k * dt
        trajectory[:, k] = ship.forward(dt)

    # Plots
    plt.figure(1, figsize=(mf.cm2inch(fig_size[0]), mf.cm2inch(fig_size[1])), dpi=dpi_value)
    plt.plot(waypoints[1], waypoints[0], "rx", label="Waypoints")
    plt.plot(trajectory[1, :], trajectory[0, :], "b", label="Trajectory")
    plt.xlabel("South (m)")
    plt.ylabel("North (m)")
    plt.legend()
    plt.grid()

    plt.figure(2, figsize=(mf.cm2inch(fig_size[0]), mf.cm2inch(fig_size[1])), dpi=dpi_value)
    plt.plot(time, trajectory[3], label="Speed")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid()
    plt.legend()

    plt.show()
    print("Done")
