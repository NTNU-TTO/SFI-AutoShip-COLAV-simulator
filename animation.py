import matplotlib.animation as animation
from map import *


def visualize(data, wp_number, t):
    fig1, ax1 = plt.subplots(figsize=(12, 10), facecolor=(0.8, 0.8, 0.8))
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
                ax1.scatter(waypoints[w][1], waypoints[w][0], color=c, s=15, alpha=0.3)
                ax1.plot([waypoints[w][1], waypoints[w + 1][1]], [waypoints[w][0], waypoints[w + 1][0]], "--"+c, alpha=0.4)

    # animation set up
    artists = circles + lines

    def init():
        ax1.set_xlim(x_lim[0] + 900, x_lim[1] - 900)
        ax1.set_ylim(y_lim[0] + 600, y_lim[1] - 600)
        return artists

    def update(i):
        for j in range(len(data)):
            circles[j].set_xdata(data[f'Ship{j+1}'][1][i])
            circles[j].set_ydata(data[f'Ship{j+1}'][0][i])
            lines[j].set_data([data[f'Ship{j+1}'][1][i], data[f'Ship{j+1}'][3][i]], [data[f'Ship{j+1}'][0][i], data[f'Ship{j+1}'][2][i]])
        artists = circles + lines
        return artists

    # display ship names
    plt.legend(loc='upper right')

    #ani = animation.FuncAnimation(fig1, update,  frames=len(t) - 1, init_func=init, blit=True, interval)
    anim = animation.FuncAnimation(fig1, update, 
                            init_func=init, 
                            frames=len(t) - 1,
                            interval = 200*t[1], 
                            blit = True) 
    plt.show()