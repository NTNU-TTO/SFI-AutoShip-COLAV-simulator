import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.plotters as plotters
import colav_simulator.core.controllers as controllers
import colav_simulator.core.guidances as guidances
import colav_simulator.core.models as models
import colav_simulator.core.sensing as sensorss
import colav_simulator.core.ship as ship
import colav_simulator.core.stochasticity as stochasticity
import colav_simulator.core.tracking.trackers as trackers
import numpy as np
from colav_simulator.scenario_generator import ScenarioGenerator
from matplotlib import pyplot as plt

legend_size = 10  # legend size
fig_size = [25, 13]  # figure1 size in cm
dpi_value = 150  # figure dpi value

if __name__ == "__main__":
    horizon = 100.0
    dt = 0.05

    utm_zone = 33
    map_size = [1500.0, 1500.0]
    map_origin_enu = [-31824.0, 6573700.0]
    map_data_files = ["Rogaland_utm33.gdb"]

    # Put new_data to True to load map data in ENC if it is not already loaded
    scenario_generator = ScenarioGenerator(
        init_enc=True, new_data=False, utm_zone=utm_zone, size=map_size, origin=map_origin_enu, files=map_data_files
    )
    origin = scenario_generator.enc_origin

    model = models.Telemetron()
    ctrl_params = controllers.FLSCParams(
        K_p_u=3.0,
        K_i_u=0.3,
        K_p_chi=2.5,
        K_d_chi=3.0,
        K_i_chi=0.1,
        max_speed_error_int=4.0,
        speed_error_int_threshold=0.5,
        max_chi_error_int=90.0 * np.pi / 180.0,
        chi_error_int_threshold=20.0 * np.pi / 180.0,
    )
    controller = controllers.FLSC(model.params, ctrl_params)
    sensor_list = [sensorss.Radar()]
    tracker = trackers.KF(sensor_list=sensor_list)
    guidance_params = guidances.LOSGuidanceParams(
        K_p=0.01,
        K_i=0.0004,
        R_a=80.0,
        max_cross_track_error_int=500.0,
        cross_track_error_int_threshold=100.0,
        pass_angle_threshold=90.0,
    )
    guidance_method = guidances.LOSGuidance(guidance_params)

    ownship = ship.Ship(
        mmsi=1,
        identifier=0,
        model=model,
        controller=controller,
        tracker=tracker,
        sensors=sensor_list,
        guidance=guidance_method,
    )

    scenario_generator.seed(1)
    csog_state = scenario_generator.generate_random_csog_state(
        draft=ownship.draft, min_land_clearance=100.0, U_min=2.0, U_max=ownship.max_speed
    )
    csog_state = np.array(
        [6574229.448438326, -31157.753734698883, 5.805685027679189, -131.1969202238676 * np.pi / 180.0]
    )
    ownship.set_initial_state(csog_state)

    rng = np.random.default_rng(seed=1)
    enc = scenario_generator.enc
    safe_sea_cdt = scenario_generator.safe_sea_cdt
    safe_sea_cdt_weights = scenario_generator.safe_sea_cdt_weights
    scenario_generator.behavior_generator.initialize_data_structures(n_ships=1)
    scenario_generator.behavior_generator.setup(
        rng, [ownship], [True], enc, safe_sea_cdt, safe_sea_cdt_weights, horizon, show_plots=True
    )

    n_wps = 4
    waypoints, _ = scenario_generator.behavior_generator.generate_random_waypoints(
        rng, x=csog_state[0], y=csog_state[1], psi=csog_state[3], draft=ownship.draft, n_wps=n_wps
    )
    speed_plan = 4.0 * np.ones(
        waypoints.shape[1]
    )  # = scenario_generator.generate_random_speed_plan(U=5.0, n_wps=waypoints.shape[1])
    ownship.set_nominal_plan(waypoints=waypoints, speed_plan=speed_plan)

    disturbance_config = stochasticity.Config()
    disturbance = stochasticity.Disturbance(disturbance_config)
    # disturbance._currents = None
    # disturbance._wind = None

    n_x, n_u = model.dims
    n_r = 9
    n_samples = round(horizon / dt)
    disturbances = np.zeros((10, n_samples))
    trajectory = np.zeros((n_x, n_samples))
    refs = np.zeros((n_r, n_samples))
    tau = np.zeros((n_u, n_samples))
    time = np.zeros(n_samples)

    map_orig = np.array([6574234.5638, -31158.2928])
    traj = np.array(
        [
            [
                -10.8436,
                -12.1109,
                -14.1172,
                -19.7815,
                -29.9166,
                -44.1482,
                -61.6429,
                -81.5254,
                -103.0452,
                -125.6093,
                -148.7625,
                -168.9708,
                -182.7403,
                -190.1094,
                -193.2888,
                -195.3983,
                -197.6000,
            ],
            [
                -24.8234,
                -52.4508,
                -80.0572,
                -107.1298,
                -132.8302,
                -156.4532,
                -177.6805,
                -196.5452,
                -213.3069,
                -228.3423,
                -242.0761,
                -253.1981,
                -260.4683,
                -264.2923,
                -265.9332,
                -267.0207,
                -267.9646,
            ],
            [
                -1.6413,
                -1.5919,
                -1.6947,
                -1.8594,
                -2.0335,
                -2.1926,
                -2.3277,
                -2.4373,
                -2.5225,
                -2.5852,
                -2.6273,
                -2.6509,
                -2.6615,
                -2.6648,
                -2.6656,
                -2.6656,
                -2.8070,
            ],
            [
                5.5268,
                5.5369,
                5.5396,
                5.5364,
                5.5282,
                5.5149,
                5.4964,
                5.4722,
                5.4420,
                5.4056,
                5.3632,
                3.8637,
                2.3647,
                0.9561,
                0.4750,
                0.4744,
                0.4846,
            ],
            [
                28.5692,
                56.4046,
                84.2553,
                112.1301,
                140.0287,
                167.9385,
                195.8336,
                223.6731,
                251.4019,
                278.9517,
                306.2435,
                328.1394,
                341.0805,
                347.5990,
                350.4900,
                352.4344,
                354.3136,
            ],
            [
                5.5662,
                5.5680,
                5.5723,
                5.5776,
                5.5818,
                5.5822,
                5.5759,
                5.5599,
                5.5316,
                5.4883,
                5.4284,
                3.3300,
                1.8464,
                0.7610,
                0.3954,
                0.3824,
                0.3693,
            ],
        ]
    )
    traj[:2, :] += map_orig.reshape(2, 1)
    state = np.array([6574226.0000, -31169.2031, -1.8932, 5.5457, 0.0, 0.0], dtype=np.float32)
    ownship._state = state

    ref_counter = 1
    t_prev_upd = 0.0
    chi_ref = traj[2, 1]
    U_ref = traj[3, 1]
    for k in range(n_samples):
        time[k] = k * dt
        disturbance_data = disturbance.get()
        if disturbance_data.wind:
            disturbances[0, k] = disturbance_data.wind["speed"]
            disturbances[1, k] = disturbance_data.wind["direction"]
        if disturbance_data.currents:
            disturbances[2, k] = disturbance_data.currents["speed"]
            disturbances[3, k] = disturbance_data.currents["direction"]

        # ownship.plan(time[k], dt, [], None, w=disturbance_data)
        if time[k] - t_prev_upd >= 5.0:
            ref_counter += 1 if ref_counter < traj.shape[1] - 1 else 0
            t_prev_upd = time[k]
            chi_ref = traj[2, ref_counter]
            U_ref = traj[3, ref_counter]

        ownship.set_references(np.array([0.0, 0.0, chi_ref, U_ref, 0.0, 0.0, 0.0, 0.0, 0.0]))
        trajectory[:, k], tau[:, k], refs[:, k] = ownship.forward(dt, w=disturbance_data)
        disturbance.update(time[k], dt)

    # Plots
    scenario_generator.enc.start_display()
    if disturbance_data.currents and disturbance_data.currents["speed"] > 0.0:
        plotters.plot_disturbance(
            magnitude=100.0,
            direction=disturbance_data.currents["direction"],
            name="current: " + str(disturbance_data.currents["speed"]) + " m/s",
            enc=enc,
            color="white",
            linewidth=1.0,
            location="topright",
            text_location_offset=(0.0, 0.0),
        )

    if disturbance_data.wind and disturbance_data.wind["speed"] > 0.0:
        plotters.plot_disturbance(
            magnitude=100.0,
            direction=disturbance_data.wind["direction"],
            name="wind: " + str(disturbance_data.wind["speed"]) + " m/s",
            enc=enc,
            color="peru",
            linewidth=1.0,
            location="topright",
            text_location_offset=(0.0, -20.0),
        )
    plotters.plot_waypoints(
        traj[:2, :],
        scenario_generator.enc,
        "orange",
        point_buffer=2.0,
        disk_buffer=4.0,
        hole_buffer=2.0,
        alpha=0.6,
    )
    plotters.plot_trajectory(trajectory, scenario_generator.enc, "black")
    for k in range(0, n_samples, 20):
        ship_poly = mapf.create_ship_polygon(
            trajectory[0, k], trajectory[1, k], trajectory[2, k], ownship.length, ownship.width
        )
        scenario_generator.enc.draw_polygon(ship_poly, "magenta", fill=True)

    # States
    fig = plt.figure(figsize=(mf.cm2inch(fig_size[0]), mf.cm2inch(fig_size[1])), dpi=dpi_value)
    axs = fig.subplot_mosaic(
        [
            ["xy", "chi", "r"],
            ["U", "u", "v"],
        ]
    )
    axs["xy"].plot(traj[1, :] - origin[1], traj[0, :] - origin[0], "rx", label="Waypoints")
    axs["xy"].plot(trajectory[1, :] - origin[1], trajectory[0, :] - origin[0], "k", label="Trajectory")
    axs["xy"].set_xlabel("East (m)")
    axs["xy"].set_ylabel("North (m)")
    axs["xy"].grid()
    axs["xy"].legend()

    axs["u"].plot(time, refs[3], "r--", label="Surge reference")
    axs["u"].plot(time, trajectory[3], "k", label="Surge")
    axs["u"].set_xlabel("Time (s)")
    axs["u"].set_ylabel("North (m)")
    axs["u"].grid()
    axs["u"].legend()

    axs["v"].plot(time, refs[4], "r--", label="Sway reference")
    axs["v"].plot(time, trajectory[4], "k", label="Sway")
    axs["v"].set_xlabel("Time (s)")
    axs["v"].set_ylabel("East (m)")
    axs["v"].grid()
    axs["v"].legend()

    # heading_error = mf.wrap_angle_diff_to_pmpi(refs[2, :], trajectory[2, :])
    crab = np.arctan2(trajectory[4, :], trajectory[3, :])
    axs["chi"].plot(time, np.rad2deg(mf.wrap_angle_to_pmpi(refs[2, :])), "r--", label="Course reference")
    axs["chi"].plot(time, np.rad2deg(mf.wrap_angle_to_pmpi(trajectory[2, :] + crab)), "k", label="Course")
    axs["chi"].set_xlabel("Time (s)")
    axs["chi"].set_ylabel("Course (deg)")
    axs["chi"].grid()
    axs["chi"].legend()

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

    # Disturbances
    fig = plt.figure(figsize=(mf.cm2inch(fig_size[0]), mf.cm2inch(fig_size[1])), dpi=dpi_value)
    axs = fig.subplot_mosaic([["wind_speed", "wind_direction"], ["current_speed", "current_direction"]])

    axs["wind_speed"].plot(time, disturbances[0, :], "k", label="Wind speed")
    axs["wind_speed"].set_xlabel("Time (s)")
    axs["wind_speed"].set_ylabel("Speed (m/s)")
    axs["wind_speed"].grid()
    axs["wind_speed"].legend()

    axs["wind_direction"].plot(time, np.rad2deg(disturbances[1, :]), "k", label="Wind direction")
    axs["wind_direction"].set_xlabel("Time (s)")
    axs["wind_direction"].set_ylabel("Direction (deg)")
    axs["wind_direction"].grid()
    axs["wind_direction"].legend()

    axs["current_speed"].plot(time, disturbances[2, :], "k", label="Current speed")
    axs["current_speed"].set_xlabel("Time (s)")
    axs["current_speed"].set_ylabel("Speed (m/s)")
    axs["current_speed"].grid()
    axs["current_speed"].legend()

    axs["current_direction"].plot(time, np.rad2deg(disturbances[3, :]), "k", label="Current direction")
    axs["current_direction"].set_xlabel("Time (s)")
    axs["current_direction"].set_ylabel("Direction (deg)")
    axs["current_direction"].grid()
    axs["current_direction"].legend()

    # Inputs
    if n_u == 3:
        fig = plt.figure(figsize=(mf.cm2inch(fig_size[0]), mf.cm2inch(fig_size[1])), dpi=dpi_value)
        axs = fig.subplot_mosaic(
            [
                ["X"],
                ["Y"],
                ["N"],
            ]
        )
        axs["X"].plot(time, tau[0, :], "k", label="Surge force")
        axs["X"].set_xlabel("Time (s)")
        axs["X"].set_ylabel("Force (N)")
        axs["X"].grid()
        axs["X"].legend()

        axs["Y"].plot(time, tau[1, :], "k", label="Sway force")
        axs["Y"].set_xlabel("Time (s)")
        axs["Y"].set_ylabel("Force (N)")
        axs["Y"].grid()
        axs["Y"].legend()

        axs["N"].plot(time, tau[2, :], "k", label="Yaw moment")
        axs["N"].set_xlabel("Time (s)")
        axs["N"].set_ylabel("Moment (Nm)")
        axs["N"].grid()
        axs["N"].legend()

    plt.show()
    print("Done")
