import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functions import *


def main():
    ###############################################
    # INITIALISATION
    ###############################################

    # time
    t0 = 0
    t_end = 60
    dt = 1
    t = np.arange(t0, t_end + dt, dt)

    #waypoints
    w1 = waypoint(0,0)
    w2 = waypoint(150,290)
    w3 = waypoint(400, 100)
    w4 = waypoint(200, -200)
    waypoints = [w1, w2, w3, w4]

    # creating ships
    ship1 = Ship(w1.x, w1.y, 1, 0, 'Ship A') #

    magnitude, angles, vectors = waypoint_vectors(ship1, waypoints)



    # creating ships data
    x1, y1, x1_t, y1_t = [], [], [], []

    ship1.c += angles[0]
    for i in range(len(t)):
        print(ship1.x, ship1.y)
        ship1.move(dt)
        x1.append(int(ship1.x))
        y1.append(int(ship1.y))
        ship1.future_pos(5)
        x1_t.append(int(ship1.x_t))
        y1_t.append(int(ship1.y_t))





    ###############################################
    # ANIMATION PART
    ###############################################

    # template
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set(xlim=(-500, 500), ylim=(-500, 500))

    ship1_scat = ax.scatter(x1, y1, color='b', s=40)
    ship1_vec, = ax.plot([], [], color='b')

    w1_scat = ax.scatter(w1.x, w1.y, color = 'r', s = 20)
    w2_scat = ax.scatter(w2.x, w2.y, color='r', s=20)
    w3_scat = ax.scatter(w3.x, w3.y, color='r', s=20)
    w4_scat = ax.scatter(w4.x, w4.y, color='r', s=20)

    origin = np.array([[w1.x,w1.y], [w2.x, w2.y]])



    # animation function
    def animate(i):
        # ship1 movement
        ship1_scat.set_offsets(np.c_[x1[i], y1[i]])
        ship1_vec.set_data([x1[i], x1_t[i]], [y1[i], y1_t[i]])

        w1_scat
        w2_scat
        w3_scat
        w4_scat





    # running the animation
    anim = animation.FuncAnimation(fig, animate, interval=100, frames=len(t) - 1)
    plt.show()





if __name__ == '__main__':
    main()
