import math
import random
import numpy as np
from matplotlib.patches import Ellipse, Circle
import shapely
from shapely.geometry import Point, Polygon, LineString, GeometryCollection
from map import *
from sensors import *


class Ship:
    '''
    x, y : Initial x and y values of the ship.
    speed: Initial speed value.
    heading: Initial true heading or course of the ship. It is a value between 0-359 degrees.
    name: Name of the ship.
    x_t, y_t: x and y position in t future. They are used for course and speed visualization vector.
    noise: Used to create a random value for ship to be off the route.
    '''
    def __init__(self, x, y, speed, heading, length, draft, mmsi, sensors=None):
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
        self.los_angle = 0

        self.a_max = os_max_acc
        self.r_rate_max = os_max_turn_rate
        self.u_max = os_max_speed

        noise = random.uniform(0, 10)
        self.wp = [(self.x + noise, self.y + noise)]
        self.idx_next_wp = 1
        if sensors is None:
            sensors = [Radar(4,0,0,0.01*np.eye(4)), AIS(6,0,0.01,0.01*np.eye(4))] # to test code before sensor params defined

        self.estimator = Estimator(sensors)     # estimates the state ([x, y, SOG, COG]) of target ship(s)
        self.target_ship_state_est = []       # list of current target ship(s) state estimate
        for i in range(ship_num-1):
            self.target_ship_state_est.append(np.zeros(4))

        # Message number 1/2/3 for ships with AIS Class A, 18 for AIS Class B ships
        # AIS Class A is required for ships bigger than 300 GT. Approximately 45 m x 10 m ship would be 300 GT.
        if self.length <= 45:
            self.message_nr = 18
        else:
            self.message_nr = random.randint(1, 3)

    def get_full_state(self):
        return np.array([self.x, self.y, self.psi, self.u, self.v, self.r])
    def get_pose_and_speed(self):
        return np.array([self.x, self.y, self.u, self.psi])

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
            self.u = self.u - 0.5 * dt
            if self.u <= 0:
                self.u = 0
        # check if the ship's position is 50 meters range of the last waypoint. If so, stop the ship.
        elif np.sqrt((self.wp[-1][0]-self.x)**2 + (self.wp[-1][1]-self.y)**2) <= 50:
            self.u = self.u - 0.5 * dt
            if self.u <= 0:
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

    def update_los_angle(self):
        self.los_angle = math.degrees(math.atan2((self.wp[self.idx_next_wp][0] - self.x),
                                                    (self.wp[self.idx_next_wp][1] - self.y)))
        if self.los_angle < 0:
            self.los_angle = 360 + self.los_angle

    def follow_waypoints(self, dt):
        # check if the ship has reached the next waypoint (within 50m range)
        if np.sqrt((self.wp[self.idx_next_wp][0]-self.x)**2 + (self.wp[self.idx_next_wp][1]-self.y)**2) <= 50:
            self.idx_next_wp = min(self.idx_next_wp+1, len(self.wp)-1)
        
        self.update_los_angle()
        delta_psi = math.atan2(np.sin(math.radians(self.los_angle)-self.psi), np.cos(math.radians(self.los_angle)-self.psi))

        # Ships turn radius is simulated according to its size:
        if self.length >= 100:
            self.psi += np.sign(delta_psi) * min(0.08 * abs(delta_psi), self.r_rate_max) * dt
        elif 25 <= self.length < 100:
            self.psi += np.sign(delta_psi) * min(0.1 * abs(delta_psi), self.r_rate_max) * dt
        elif self.length < 25:
            self.psi += np.sign(delta_psi) * min(0.2 * abs(delta_psi), self.r_rate_max) * dt

    def future_pos(self, time):
        # Future position in defined time will be used to visualize ship's heading
        self.x_t = self.x - self.u * math.sin(-self.psi) * time
        self.y_t = self.y + self.u * math.cos(-self.psi) * time

    def update_target_x_est(self, x_true_list, t, dt):
        """
            Updates the state estimates of the other ships.
            x_true_list: list of true states of target ships,
            the order of the ships must match in x_true_list and self.target_ship_state_est 
        """
        for i, x_est in enumerate(self.target_ship_state_est):
            if t == 0:
                self.target_ship_state_est[i] = x_true_list[i] #initial state estimates set to true value
            else:
                self.target_ship_state_est[i] = self.estimator.step(x_true_list[i], x_est, t, dt)

    def get_converted_target_x_est(self):
        """
            Returns list of target ship estimates in the form [x, y, Vx, Vy]
        """
        x_est_list = []
        for i in range(len(self.target_ship_state_est)):
            x_est = np.zeros(4)
            x_est[0:2] = self.target_ship_state_est[i][0:2]
            x_est[2] = self.target_ship_state_est[i][2]*np.sin(self.target_ship_state_est[i][3])
            x_est[3] = self.target_ship_state_est[i][2]*np.cos(self.target_ship_state_est[i][3])
            x_est_list.append(x_est)
        return x_est_list


    

    
