
import colav_simulator.utils.math_functions as mf
import numpy as np
from colav_simulator.ships.ship import Ship
from matplotlib import pyplot as plt

legend_size = 10  # legend size
fig_size = [25, 13]  # figure1 size in cm
dpi_value = 150  # figure dpi value

if __name__ == '__main__':

    waypoints = np.array([[0, 50, 100, 2000], [0, 50, 100, 2000]])
    speed_plan = np.array([6.0, 6.0, 6.0, 6.0])
    ship = Ship(waypoints, speed_plan)

    horizon = 200.0
    dt = 1.0
    n = 4 # when using kinematic model considering only pos and vel
    n_samples = round(horizon / dt)
    trajectory = np.zeros([n, n_samples])
    time =  np.zeros([1, n_samples])
    for k in range(n_samples):
        time[k] = k * dt
        trajectory[k] = ship.forwardd(dt)


    # Plots
    plt.figure(1, figsize=(mf.cm2inch(fig_size[0]), mf.cm2inch(fig_size[1])), dpi=dpi_value)
    plt.grid()

    plt.subplot(2, 1, 1)
    plt.plot(trajectory[1, :], trajectory[0, :])
    plt.legend(["North-East positions (m)"], fontsize=legend_size)
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time, trajectory[3])
    plt.legend(["Vessel speed (m/s)"], fontsize = legend_size)
    plt.grid()

    plt.show()
