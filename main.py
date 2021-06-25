import matplotlib.animation as animation
from scenario_generator import *
from map import *



def main():
    ###############################################
    # INITIALISATION
    ###############################################

    # time
    t = np.arange(time_start, time_end + time_step, time_step)

    # number of waypoints
    wp_number = waypoint_num

    # scenarios
    ship_list = ship_generator(Ship, scenario_num=scenario_num,
                               os_max_speed=os_max_speed, ts_max_speed=ts_max_speed, ship_number=ship_num)

    waypoint_list = waypoint_generator(ships=ship_list, waypoints_number=wp_number)

    data, ais_data = ship_data(ships=ship_list, waypoint_list=waypoint_list, time=t, timestep=time_step)

    # exporting ais_data.csv
    ais_data.to_csv('ais_data.csv')

    ###############################################
    # ANIMATION PART
    ###############################################

    # template of the environment
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=(0.8, 0.8, 0.8))
    x_lim, y_lim = background('show')
    ax.set(xlim=(x_lim[0]+900, x_lim[1]-900), ylim=(y_lim[0]+600, y_lim[1]-600))


    # ship visualization framework
    ship_scat_list = []
    ship_vec_list = []

    for j in range(1, len(data) + 1):
        x = data[f'Ship{j}'][0]
        y = data[f'Ship{j}'][1]
        if j == 1:
            name = 'Own ship'
            c = 'b'
        elif j == 2:
            name = 'Target ship'
            c = 'r'
        else:
            name = f'Ship {j}'
            c = 'k'
        ship_scat = ax.scatter(x, y, color=c, s=40, label=name)
        ship_vec, = ax.plot([], [], color=c)
        ship_scat_list.append(ship_scat)
        ship_vec_list.append(ship_vec)

    plt.legend(loc='upper right')

    # visualizing waypoints
    for j in range(1, len(data) + 1):
        if j == 1:
            c = 'b'
        elif j == 2:
            c = 'r'
        else:
            c = 'k'
        waypoints = data[f'Ship{j}'][4]
        for w in range(0, wp_number):
            ax.scatter(waypoints[w][0], waypoints[w][1], color=c, s=20, alpha=0.4)
            ax.plot([waypoints[w][0], waypoints[w + 1][0]], [waypoints[w][1], waypoints[w + 1][1]], "--"+c,
                    alpha=0.3)

    # animation function
    def animate(i):
        for ix in range(1, len(data) + 1):
            ship_scat_list[ix - 1].set_offsets(np.c_[data[f'Ship{ix}'][0][i], data[f'Ship{ix}'][1][i]])
            ship_vec_list[ix - 1].set_data([data[f'Ship{ix}'][0][i], data[f'Ship{ix}'][2][i]], [data[f'Ship{ix}'][1][i],
                                                                                                data[f'Ship{ix}'][3][
                                                                                                    i]])

    # running the animation
    anim = animation.FuncAnimation(fig, animate, interval=100, frames=len(t) - 1)
    plt.show()


if __name__ == '__main__':
    main()
