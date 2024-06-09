"""
    Test module for the Ship class. Use it to test the ship's behavior in a scenario, and/or tune your specific guidance algorithm + controller.
"""

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
from colav_simulator.scenario_config import OwnshipPositionGenerationMethod
from colav_simulator.scenario_generator import ScenarioGenerator
from matplotlib import pyplot as plt

legend_size = 10  # legend size
fig_size = [25, 13]  # figure1 size in cm
dpi_value = 150  # figure dpi value

if __name__ == "__main__":
    horizon = 100.0
    dt = 0.1  # NOTE: time step affects the dynamics accuracy and also control performance

    utm_zone = 33
    map_size = [1500.0, 1500.0]
    map_origin_enu = [-31824.0, 6573700.0]
    map_data_files = ["Rogaland_utm33.gdb"]

    # Put new_data to True to load map data in ENC if it is not already loaded
    scenario_generator = ScenarioGenerator(
        init_enc=True, new_data=True, utm_zone=utm_zone, size=map_size, origin=map_origin_enu, files=map_data_files
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
        method=OwnshipPositionGenerationMethod.UniformInTheMapThenGaussian,
        draft=ownship.draft,
        min_hazard_clearance=100.0,
        U_min=2.0,
        U_max=ownship.max_speed,
    )
    csog_state = np.array(
        [6574229.448438326, -31157.753734698883, 5.805685027679189, -131.1969202238676 * np.pi / 180.0]
    )
    ownship.set_initial_state(csog_state)

    disturbance_config = stochasticity.Config()
    disturbance = stochasticity.Disturbance(disturbance_config)
    # disturbance._currents = None
    # disturbance._wind = None

    rng = np.random.default_rng(seed=1)
    enc = scenario_generator.enc
    safe_sea_cdt = scenario_generator.safe_sea_cdt
    safe_sea_cdt_weights = scenario_generator.safe_sea_cdt_weights
    scenario_generator.behavior_generator.initialize_data_structures(1)
    scenario_generator.behavior_generator.setup_enc(
        enc=enc, safe_sea_cdt=safe_sea_cdt, safe_sea_cdt_weights=safe_sea_cdt_weights
    )
    scenario_generator.behavior_generator.setup_ship(
        rng=rng, ship_obj=ownship, replan=True, simulation_timespan=horizon, show_plots=True
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

    map_orig = np.array([6574512.2536, -31378.5372])
    traj = np.array(
        [
            [
                81.5652,
                104.7765,
                124.5801,
                136.6765,
                139.8153,
                138.2139,
                136.6566,
                135.3308,
                134.3252,
                133.6635,
                133.3588,
                133.2365,
                133.1173,
                133.0005,
                132.8854,
                132.7711,
                132.1291,
            ],
            [
                -139.6983,
                -169.3614,
                -201.2551,
                -236.7254,
                -274.0106,
                -311.4461,
                -346.3892,
                -375.1232,
                -396.3719,
                -410.1322,
                -416.4141,
                -418.9215,
                -421.3578,
                -423.7392,
                -426.0842,
                -428.4122,
                -430.6641,
            ],
            [
                -0.9066,
                -0.9070,
                -1.1233,
                -1.3611,
                -1.6126,
                -1.6145,
                -1.6162,
                -1.6176,
                -1.6186,
                -1.6191,
                -1.6194,
                -1.6196,
                -1.6198,
                -1.6198,
                -1.6199,
                -1.6199,
                -2.0750,
            ],
            [
                7.5379,
                7.5281,
                7.5179,
                7.5079,
                7.4984,
                7.4895,
                6.5016,
                5.0042,
                3.5048,
                2.0057,
                0.5100,
                0.4942,
                0.4815,
                0.4723,
                0.4669,
                0.4655,
                0.4793,
            ],
            [
                153.0425,
                190.2473,
                227.4443,
                264.6259,
                301.7756,
                338.8675,
                374.2344,
                405.2686,
                427.1566,
                438.7751,
                443.8346,
                445.9798,
                447.9524,
                449.8519,
                451.7122,
                453.3884,
                454.7265,
            ],
            [
                7.4416,
                7.4403,
                7.4384,
                7.4342,
                7.4257,
                7.4111,
                6.7357,
                5.6780,
                3.0772,
                1.5702,
                0.4537,
                0.4044,
                0.3847,
                0.3751,
                0.3690,
                0.3014,
                0.2338,
            ],
        ]
    )
    traj[:2, :] += map_orig.reshape(2, 1)
    state = np.array([6574583.5000, -31501.8203, -1.1510, 7.4657, -0.8312, 0.2618], dtype=np.float32)
    ownship._state = state

    ref_counter = 0
    t_prev_upd = 0.0
    chi_ref = traj[2, 1]
    U_ref = traj[3, 1]
    for k in range(n_samples):
        time[k] = k * dt
        disturbance_data = disturbance.get()
        disturbance_data.currents = {"speed": 1.0429484, "direction": 0.59327924}
        disturbance_data.wind = {"speed": 6.546729, "direction": -3.7480865}
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
    for k in range(0, n_samples, 40):
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

    plt.show(block=False)
    print("Done")
