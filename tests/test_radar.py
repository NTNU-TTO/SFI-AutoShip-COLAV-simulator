import colav_simulator.core.sensing as sensing
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def test_radar() -> None:

    rparams = sensing.RadarParams()
    rparams.generate_clutter = True
    rparams.max_range = 100.0
    radar = sensing.Radar(rparams)

    ownship_state = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    true_do_states = [
        (0, np.array([50.0, 0.0, -5.0, 0.0]), 10.0, 3.0)
    ]  # dynamic obstacle info on the form (ID, state, length, width)
    t = 1.0

    matplotlib.use("TkAgg")
    fig, ax = plt.subplots()
    plt.ion()
    plt.show(block=False)
    ax.plot(ownship_state[1], ownship_state[0], "bx")
    ax.plot(true_do_states[0][1][1], true_do_states[0][1][0], "mx")
    for k in range(5):
        t = k * 1.0
        meas = radar.generate_measurements(t, true_do_states, ownship_state)

        for m in meas:
            if np.any(np.isnan(m[1])):
                continue
            ax.plot(m[1][1], m[1][0], "ro")

    rparams.generate_clutter = False
    radar = sensing.Radar(rparams)
    for k in range(5):
        t = k * 1.0
        meas = radar.generate_measurements(t, true_do_states, ownship_state)

        for m in meas:
            if np.any(np.isnan(m[1])):
                continue
            ax.plot(m[1][1], m[1][0], "go")

    rparams.include_polar_meas_noise = True
    radar = sensing.Radar(rparams)
    for k in range(5):
        t = k * 1.0
        meas = radar.generate_measurements(t, true_do_states, ownship_state)

        for m in meas:
            if np.any(np.isnan(m[1])):
                continue
            ax.plot(m[1][1], m[1][0], "ko")

    ax.set_aspect("equal")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_xlim(-rparams.max_range, rparams.max_range)
    ax.set_ylim(-rparams.max_range, rparams.max_range)

    print("done")


if __name__ == "__main__":
    test_radar()
