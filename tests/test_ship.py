import colav_simulator.ships.ship as ship
import colav_simulator.utils.math_functions as mf
import numpy as np
from matplotlib import pyplot as plt

legend_size = 10  # legend size
fig_size = [25, 13]  # figure1 size in cm
dpi_value = 150  # figure dpi value

if __name__ == "__main__":
    n_wps = 10
    waypoints = np.zeros((2, n_wps))
    for i in range(n_wps):
        if i == 0:
            waypoints[:, i] = np.array([0, 0])
        elif i == 1 or i == 2:
            waypoints[:, i] = waypoints[:, i - 1] + np.array([51, 50])
        elif i > 2 and i <= 4:
            waypoints[:, i] = waypoints[:, i - 1] + np.array([-50, 50])
        elif i > 4 and i <= 6:
            waypoints[:, i] = waypoints[:, i - 1] + np.array([-50, -50])
        elif i > 6:
            waypoints[:, i] = waypoints[:, i - 1] + np.array([50, -50])

    speed_plan = np.array([6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0])
    state = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    mmsi = "lol"
    ownship = ship.Ship(mmsi, waypoints, speed_plan, state)

    horizon = 4.0
    dt = 0.1
    n = 6  # n = 4 when using kinematic model considering only pos and vel, and 6 otherwise
    n_samples = round(horizon / dt)
    trajectory = np.zeros((n, n_samples))
    time = np.zeros(n_samples)
    for k in range(n_samples):
        time[k] = k * dt
        trajectory[:, k] = ownship.forward(dt)

    # Plots
    plt.figure(1, figsize=(mf.cm2inch(fig_size[0]), mf.cm2inch(fig_size[1])), dpi=dpi_value)
    plt.plot(waypoints[1, :], waypoints[0, :], "rx", label="Waypoints")
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
