import colav_simulator.common.math_functions as mf
import colav_simulator.core.ship as ship
import numpy as np
from colav_simulator.scenario_management import ScenarioGenerator
from matplotlib import pyplot as plt

legend_size = 10  # legend size
fig_size = [25, 13]  # figure1 size in cm
dpi_value = 150  # figure dpi value

if __name__ == "__main__":

    n_wps = 9
    scenario_generator = ScenarioGenerator()
    origin = scenario_generator.enc_origin

    # waypoints = np.zeros((2, n_wps))
    # for i in range(n_wps):
    #     if i == 0:
    #         waypoints[:, i] = np.array([0, 0])
    #     elif i == 1:
    #         waypoints[:, i] = waypoints[:, i - 1] + np.array([50, 0])
    #     elif i >= 2 and i < 4:
    #         waypoints[:, i] = waypoints[:, i - 1] + np.array([-50, 50])
    #     elif i >= 4 and i < 6:
    #         waypoints[:, i] = waypoints[:, i - 1] + np.array([-50, -50])
    #     elif i >= 6:
    #         waypoints[:, i] = waypoints[:, i - 1] + np.array([50, -50])
    # speed_plan = np.array([6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]) / 6.0

    ownship = ship.Ship(mmsi=1)

    pose = scenario_generator.generate_random_pose(draft=ownship.draft, land_clearance=200.0)
    waypoints = scenario_generator.generate_random_waypoints(
        x=pose[0], y=pose[1], psi=pose[3], draft=ownship.draft, n_wps=n_wps
    )
    speed_plan = scenario_generator.generate_random_speed_plan(U=5.0, n_wps=waypoints.shape[1])

    ownship.set_initial_state(pose)
    ownship.set_nominal_plan(waypoints=waypoints, speed_plan=speed_plan)

    horizon = 300.0
    dt = 0.1
    n_x = 6
    n_r = 9
    n_u = 3
    n_samples = round(horizon / dt)
    trajectory = np.zeros((n_x, n_samples))
    refs = np.zeros((n_r, n_samples))
    tau = np.zeros((n_u, n_samples))
    time = np.zeros(n_samples)
    for k in range(n_samples):
        time[k] = k * dt
        trajectory[:, k], tau[:, k], refs[:, k] = ownship.forward(dt)

    # Plots
    gcf = plt.gcf()
    gca = gcf.axes[0]
    gca.plot(waypoints[1, :], waypoints[0, :], "rx", label="Waypoints")
    gca.plot(trajectory[1, :], trajectory[0, :], "k", label="Trajectory")
    gca.set_xlabel("South (m)")
    gca.set_ylabel("North (m)")
    gca.legend()
    gca.grid()

    fig = plt.figure(figsize=(mf.cm2inch(fig_size[0]), mf.cm2inch(fig_size[1])), dpi=dpi_value)
    axs = fig.subplot_mosaic(
        [
            ["xy", "psi", "r"],
            ["U", "x", "y"],
        ]
    )

    axs["x"].plot(time, refs[0] - trajectory[0], label="Tracking error north")
    axs["x"].set_xlabel("Time (s)")
    axs["x"].set_ylabel("North (m)")
    axs["x"].grid()
    axs["x"].legend()

    axs["y"].plot(time, refs[1] - trajectory[1], label="Tracking error east")
    axs["y"].set_xlabel("Time (s)")
    axs["y"].set_ylabel("East (m)")
    axs["y"].grid()
    axs["y"].legend()

    heading_error = mf.wrap_angle_diff_to_pmpi(refs[2, :], trajectory[2, :])

    axs["psi"].plot(time, np.rad2deg(heading_error), label="Heading error")
    axs["psi"].set_xlabel("Time (s)")
    axs["psi"].set_ylabel("Heading (deg)")
    axs["psi"].grid()
    axs["psi"].legend()

    axs["r"].plot(time, np.rad2deg(refs[5] - trajectory[5]), label="Yaw rate error")
    axs["r"].set_xlabel("Time (s)")
    axs["r"].set_ylabel("Angular rate rate (deg/s)")
    axs["r"].grid()
    axs["r"].legend()

    U_d = np.sqrt(refs[3] ** 2 + refs[4] ** 2)
    U = np.sqrt(trajectory[3, :] ** 2 + trajectory[4, :] ** 2)
    axs["U"].plot(time, U_d - U, label="Speed error")
    axs["U"].set_xlabel("Time (s)")
    axs["U"].set_ylabel("Speed (m/s)")
    axs["U"].grid()
    axs["U"].legend()

    plt.show()
    print("Done")
