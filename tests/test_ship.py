import colav_simulator.common.math_functions as mf
import colav_simulator.core.controllers as controllers
import colav_simulator.core.guidances as guidances
import colav_simulator.core.models as models
import colav_simulator.core.sensing as sensorss
import colav_simulator.core.ship as ship
import colav_simulator.core.tracking.trackers as trackers
import numpy as np
from colav_simulator.scenario_management import ScenarioGenerator
from matplotlib import pyplot as plt

legend_size = 10  # legend size
fig_size = [25, 13]  # figure1 size in cm
dpi_value = 150  # figure dpi value

if __name__ == "__main__":

    n_wps = 4

    # Put new_data to True to load map data in ENC if it is not already loaded
    scenario_generator = ScenarioGenerator(init_enc=True, new_data=False)
    scenario_generator.enc.start_display()
    origin = scenario_generator.enc_origin

    model = models.Telemetron()
    controller = controllers.FLSH()
    sensor_list = [sensorss.Radar()]
    tracker = trackers.KF(sensor_list=sensor_list)
    guidance_method = guidances.LOSGuidance()

    ownship = ship.Ship(mmsi=1, identifier=0, model=model, controller=controller, tracker=tracker, sensors=sensor_list, guidance=guidance_method)

    csog_state = scenario_generator.generate_random_csog_state(draft=ownship.draft, min_land_clearance=400.0)
    waypoints = scenario_generator.generate_random_waypoints(x=csog_state[0], y=csog_state[1], psi=csog_state[3], draft=ownship.draft, n_wps=n_wps)
    speed_plan = scenario_generator.generate_random_speed_plan(U=5.0, n_wps=waypoints.shape[1])

    csog_state[3] += 90.0 * np.pi / 180.0
    ownship.set_initial_state(csog_state)
    ownship.set_nominal_plan(waypoints=waypoints, speed_plan=speed_plan)

    # Plots
    gcf = plt.gcf()
    gca = gcf.axes[0]
    gca.plot(waypoints[1, :], waypoints[0, :], "rx", label="Waypoints")

    horizon = 400.0
    dt = 0.5
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
        ownship.plan(time[k], dt, [], None)
        trajectory[:, k], tau[:, k], refs[:, k] = ownship.forward(dt)

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

    axs["xy"].plot(waypoints[1, :], waypoints[0, :], "rx", label="Waypoints")
    axs["xy"].plot(trajectory[1, :], trajectory[0, :], "k", label="Trajectory")
    axs["xy"].set_xlabel("East (m)")
    axs["xy"].set_ylabel("North (m)")
    axs["xy"].grid()
    axs["xy"].legend()

    axs["x"].plot(time, refs[0], "r--", label="North reference")
    axs["x"].plot(time, trajectory[0], "k", label="North")
    axs["x"].set_xlabel("Time (s)")
    axs["x"].set_ylabel("North (m)")
    axs["x"].grid()
    axs["x"].legend()

    axs["y"].plot(time, refs[1], "r--", label="East reference")
    axs["y"].plot(time, trajectory[1], "k", label="East")
    axs["y"].set_xlabel("Time (s)")
    axs["y"].set_ylabel("East (m)")
    axs["y"].grid()
    axs["y"].legend()

    # heading_error = mf.wrap_angle_diff_to_pmpi(refs[2, :], trajectory[2, :])

    axs["psi"].plot(time, np.rad2deg(mf.wrap_angle_to_pmpi(refs[2, :])), "r--", label="Heading reference")
    axs["psi"].plot(time, np.rad2deg(mf.wrap_angle_to_pmpi(trajectory[2, :])), "k", label="Heading")
    axs["psi"].set_xlabel("Time (s)")
    axs["psi"].set_ylabel("Heading (deg)")
    axs["psi"].grid()
    axs["psi"].legend()

    axs["r"].plot(time, np.rad2deg(mf.wrap_angle_to_pmpi(refs[5, :])), "r--", label="Yaw rate reference")
    axs["r"].plot(time, np.rad2deg(mf.wrap_angle_to_pmpi(trajectory[5])), "k", label="Yaw rate")
    axs["r"].set_xlabel("Time (s)")
    axs["r"].set_ylabel("Angular rate rate (deg/s)")
    axs["r"].grid()
    axs["r"].legend()

    U_d = np.sqrt(refs[3] ** 2 + refs[4] ** 2)
    U = np.sqrt(trajectory[3, :] ** 2 + trajectory[4, :] ** 2)
    axs["U"].plot(time, U_d, "r--", label="Speed reference")
    axs["U"].plot(time, U, "k", label="Speed")
    axs["U"].set_xlabel("Time (s)")
    axs["U"].set_ylabel("Speed (m/s)")
    axs["U"].grid()
    axs["U"].legend()

    plt.show()
    print("Done")
