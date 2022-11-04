import colav_simulator.common.math_functions as mf
import colav_simulator.ships.ship as ship
import numpy as np
from matplotlib import pyplot as plt

legend_size = 10  # legend size
fig_size = [25, 13]  # figure1 size in cm
dpi_value = 150  # figure dpi value

if __name__ == "__main__":
    n_wps = 9
    waypoints = np.zeros((2, n_wps))
    for i in range(n_wps):
        if i == 0:
            waypoints[:, i] = np.array([0, 0])
        elif i == 1:
            waypoints[:, i] = waypoints[:, i - 1] + np.array([50, 0])
        elif i >= 2 and i < 4:
            waypoints[:, i] = waypoints[:, i - 1] + np.array([-50, 50])
        elif i >= 4 and i < 6:
            waypoints[:, i] = waypoints[:, i - 1] + np.array([-50, -50])
        elif i >= 6:
            waypoints[:, i] = waypoints[:, i - 1] + np.array([50, -50])

    speed_plan = np.array([6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]) / 6.0

    # speed_plan = np.array([6.0, 6.0, 6.0, 6.0])
    # waypoints = np.array([[0, 100, 200, 300], [0, 0, 0, 0]])

    state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    mmsi = "lol"
    ownship = ship.Ship(mmsi, waypoints, speed_plan, state)

    horizon = 200.0
    dt = 0.1
    n = 6  # n = 4 when using kinematic model considering only pos and vel, and 6 otherwise
    n_samples = round(horizon / dt)
    trajectory = np.zeros((n, n_samples))
    refs = np.zeros((9, n_samples))
    tau = np.zeros((3, n_samples))
    time = np.zeros(n_samples)
    for k in range(n_samples):
        time[k] = k * dt
        data = ownship.forward(dt)
        trajectory[:, k] = data[0:6]
        refs[:, k] = data[6:15]
        tau[:, k] = data[15:]

    # Plots
    fig = plt.figure(figsize=(mf.cm2inch(fig_size[0]), mf.cm2inch(fig_size[1])), dpi=dpi_value)
    axs = fig.subplot_mosaic(
        [
            ["xy", "psi", "r"],
            ["U", "x", "y"],
        ]
    )
    axs["xy"].plot(waypoints[1, :], waypoints[0, :], "rx", label="Waypoints")
    axs["xy"].plot(trajectory[1, :], trajectory[0, :], "b", label="Trajectory")
    axs["xy"].set_xlabel("South (m)")
    axs["xy"].set_ylabel("North (m)")
    axs["xy"].legend()
    axs["xy"].grid()

    axs["psi"].plot(time, np.rad2deg(refs[3]), label="Ref. heading")
    axs["psi"].plot(time, np.rad2deg(trajectory[3]), label="heading")
    axs["psi"].set_xlabel("Time (s)")
    axs["psi"].set_ylabel("Heading (deg)")
    axs["psi"].grid()
    axs["psi"].legend()

    axs["r"].plot(time, np.rad2deg(refs[5]), label="Ref. yaw rate")
    axs["r"].plot(time, np.rad2deg(trajectory[5]), label="yaw rate")
    axs["r"].set_xlabel("Time (s)")
    axs["r"].set_ylabel("Angular rate rate (deg/s)")
    axs["r"].grid()
    axs["r"].legend()

    axs["U"].plot(time, refs[3], label="Ref. speed")
    axs["U"].plot(time, trajectory[3], label="Speed")
    axs["U"].set_xlabel("Time (s)")
    axs["U"].set_ylabel("Speed (m/s)")
    axs["U"].grid()
    axs["U"].legend()

    axs["x"].plot(time, refs[0], label="Ref. north")
    axs["x"].plot(time, trajectory[0], label="north")
    axs["x"].set_xlabel("Time (s)")
    axs["x"].set_ylabel("North (m)")
    axs["x"].grid()
    axs["x"].legend()

    axs["y"].plot(time, refs[1], label="Ref. east")
    axs["y"].plot(time, trajectory[1], label="east")
    axs["y"].set_xlabel("Time (s)")
    axs["y"].set_ylabel("East (m)")
    axs["y"].grid()
    axs["y"].legend()

    plt.show()
    print("Done")
