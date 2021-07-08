import math
import random
import numpy as np
from matplotlib.patches import Ellipse, Circle
import shapely
from shapely.geometry import Point, Polygon, LineString, GeometryCollection
from map import *


class Ship:
    '''
    x, y : Initial x and y values of the ship.
    speed: Initial speed value.
    heading: Initial true heading or course of the ship. It is a value between 0-359 degrees.
    name: Name of the ship.
    x_t, y_t: x and y position in t future. They are used for course and speed visualization vector.
    noise: Used to create a random value for ship to be off the route.
    '''
    def __init__(self, x, y, speed, heading, length, draft, mmsi):
        self.x = x
        self.y = y
        self.psi = math.radians(heading)    # In radians
        self.u = speed * 0.51               # longitudinal velocity. Converting knots to meters per second
        self.v = 0                          # lateral velocity
        self.r = 0                          # angular velocity
        self.length = length
        self.draft = draft
        self.mmsi = mmsi
        self.x_t = 0
        self.y_t = 0

        self.a_max = os_max_acc
        self.r_rate_max = os_max_turn_rate
        self.u_max = os_max_speed

        noise = random.uniform(0, 10)
        self.wp = [(self.x + noise, self.y + noise)]
        self.idx_next_wp = 1

        # Message number 1/2/3 for ships with AIS Class A, 18 for AIS Class B ships
        # AIS Class A is required for ships bigger than 300 GT. Approximately 45 m x 10 m ship would be 300 GT.
        if self.length <= 45:
            self.message_nr = 18
        else:
            self.message_nr = random.randint(1, 3)

    def waypoints(self, wp_number):
        '''
        Creates random waypoints starting from the ship's position.
        wp_number: Number of waypoints to create.
        n: Random distance between waypoints.
        alpha: Angle in radians to make waypoints in zigzag shape.
               A bigger value makes the every odd waypoint away from the initial direction.
        '''
        for each in range(wp_number):
            n = random.randint(200, 1000)
            alpha = random.uniform(0, 0.7)
            if not each:
                alpha = 0
            wp_x = self.wp[each][0] - n * math.sin(-self.psi + alpha)
            wp_y = self.wp[each][1] + n * math.cos(-self.psi + alpha)
            # check if the waypoint path is intersecting with the shore polygon
            wp_line = LineString([(self.wp[each][0], self.wp[each][1]), (wp_x, wp_y)])
            if wp_line.intersects(enc.shore.geometry) == True:
                wp_x = self.wp[each][0]
                wp_y = self.wp[each][1]
            self.wp.append((wp_x, wp_y))

        return self.wp

    def move(self, dt):
        self.x -= self.u * math.sin(-self.psi) * dt
        self.y += self.u * math.cos(-self.psi) * dt
        # check if the ship in the future is inside the shore polygons. If so, stop the ship.
        self.future_pos(10)
        ship_pos = Point(self.x_t, self.y_t)
        if enc.shore.geometry.contains(ship_pos) == True:
            self.u = 0
        # check if the ship's position is 50 meters range of the last waypoint. If so, stop the ship.
        elif np.sqrt((self.wp[-1][0]-self.x)**2 + (self.wp[-1][1]-self.y)**2) <= 50:
            self.u = 0

        if self.u:
            """
            !!TEMP v_d = self.v!!, to be updated when speed change is required
            """
            self.update_speed(dt, u_d=self.u)

    def update_speed(self, dt, u_d):
        """
            Update speed based on simple kinematic model with constraints
        """
        a = np.sign(u_d - self.u)*min(0.1*abs(u_d - self.u), self.a_max)
        self.u += a*dt
        self.u = np.sign(self.u)*min(abs(self.u), self.u_max)


    def follow_waypoints(self, dt):
        #check if we have reached the next waypoint (within 50m range)
        if np.sqrt((self.wp[self.idx_next_wp][0]-self.x)**2 + (self.wp[self.idx_next_wp][1]-self.y)**2) <= 50:
            self.idx_next_wp = min(self.idx_next_wp+1, len(self.wp)-1)
        
        los_angle = math.degrees(math.atan2((self.wp[self.idx_next_wp][0] - self.x),
                                                    (self.wp[self.idx_next_wp][1] - self.y)))
        if los_angle < 0:
            los_angle = 360 + los_angle
        #self.psi = math.radians(los_angle)
        delta_psi = math.atan2(np.sin(math.radians(los_angle)-self.psi), np.cos(math.radians(los_angle)-self.psi))
        self.psi += np.sign(delta_psi)*min(0.1*abs(delta_psi), self.r_rate_max)*dt


    def future_pos(self, time):
        # Future position in defined time will be used to visualize ship's heading
        self.x_t = self.x - self.u * math.sin(-self.psi) * time
        self.y_t = self.y + self.u * math.cos(-self.psi) * time
