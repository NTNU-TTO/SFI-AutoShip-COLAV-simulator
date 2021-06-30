import matplotlib.animation as animation
from scenario_generator import *
from map import *
from yaspin import yaspin


@yaspin(text="Running...")
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
    fig1, ax1 = plt.subplots(figsize=(9, 6), facecolor=(0.8, 0.8, 0.8))
    x_lim, y_lim = background('show')

    # make the position(circle) and speed(line) visualizing
    circles = []
    lines = []
    for i in range(len(data)):
        if i == 0:
            c = 'b'
            circles.append(plt.plot([], [], c + 'o', label='Own ship')[0])
            lines.append(plt.plot([], [], c + '-')[0])
        elif i == 1:
            c = 'r'
            circles.append(plt.plot([], [], c + 'o', label='Target ship')[0])
            lines.append(plt.plot([], [], c + '-')[0])
        else:
            c = 'k'
            circles.append(plt.plot([], [], 'ko')[0])
            lines.append(plt.plot([], [], 'k-')[0])
        if show_waypoints:
            waypoints = data[f'Ship{i+1}'][4]
            for w in range(wp_number):
                ax1.scatter(waypoints[w][0], waypoints[w][1], color=c, s=20, alpha=0.4)
                ax1.plot([waypoints[w][0], waypoints[w + 1][0]], [waypoints[w][1], waypoints[w + 1][1]], "--"+c, alpha=0.4)


    # animation set up
    artists = circles + lines

    def init():
        ax1.set_xlim(x_lim[0] + 900, x_lim[1] - 900)
        ax1.set_ylim(y_lim[0] + 600, y_lim[1] - 600)
        return artists

    def update(i):
        for j in range(len(data)):
            circles[j].set_xdata(data[f'Ship{j+1}'][0][i])
            circles[j].set_ydata(data[f'Ship{j+1}'][1][i])
            lines[j].set_data([data[f'Ship{j+1}'][0][i], data[f'Ship{j+1}'][2][i]], [data[f'Ship{j+1}'][1][i], data[f'Ship{j+1}'][3][i]])
        artists = circles + lines
        return artists

    plt.legend(loc='upper right')
    ani = animation.FuncAnimation(fig1, update,  frames=len(t) - 1, init_func=init, blit=True)
    plt.show()


if __name__ == '__main__':
    main()



