import math
import random
import numpy as np
from matplotlib.patches import Ellipse, Circle
import shapely
from shapely.geometry import Point, Polygon, LineString, GeometryCollection
from map import *
from sensors import *
from utils import *

from read_config import read_control_config
delta, R_a = read_control_config()

class Ship:
    '''
    x, y : Initial x and y values of the ship.
    speed: Initial speed value.
    heading: Initial true heading or course of the ship. It is a value between 0-359 degrees.
    name: Name of the ship.
    x_t, y_t: x and y position in t future. They are used for course and speed visualization vector.
    noise: Used to create a random value for ship to be off the route.
    '''
    def __init__(self, x, y, speed, heading, ship_model_name, mmsi, sensors=None):
        # Choosing specific ship model
        self.ship_model = create_ship_model(ship_model_name)
        # True states and ship parameters
        self.x = x
        self.y = y
        self.psi = wrap_to_pi(math.radians(heading))    # In radians
        self.u = speed * 0.51               # longitudinal velocity. Converting knots to meters per second
        self.v = 0                          # lateral velocity
        self.r = 0                          # angular velocity
        self.length = self.ship_model.length
        self.draft = self.ship_model.draft
        self.mmsi = mmsi
        self.x_t = 0
        self.y_t = 0
        self.U_d = self.u

        # LOS guidance parameters
        self.los_angle = 0
        self.delta = delta                  # lookahead distance
        self.R_a = R_a                      # radius of acceptance

        # Waypoint parameters
        noise = random.uniform(0, 10)
        self.wp = [(self.x + noise, self.y + noise)]
        self.idx_next_wp = 1

        # Other ship state estimation vaiables
        if sensors is None:
            sensors = [Radar(4,0,0,0.01*np.eye(4)), AIS(6,0,0.01,0.01*np.eye(4))] # to test code before sensor params defined
        self.estimator = Estimator(sensors)     # estimates the state ([x, y, SOG, COG]) of target ship(s)
        self.target_ship_state_est = []       # list of current target ship(s) state estimate
        for i in range(ship_num-1):
            self.target_ship_state_est.append(np.zeros(4))

        # Message number 1/2/3 for ships with AIS Class A, 18 for AIS Class B ships
        # AIS Class A is required for ships bigger than 300 GT. Approximately 45 m x 10 m ship would be 300 GT.
        if self.ship_model.length <= 45:
            self.message_nr = 18
        else:
            self.message_nr = random.randint(1, 3)

    def get_full_state(self):
        return np.array([self.x, self.y, self.psi, self.u, self.v, self.r])

    def get_pose(self):
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
            wp_x = self.wp[each][0] + n * math.cos(self.psi + alpha)
            wp_y = self.wp[each][1] + n * math.sin(self.psi + alpha)
            # check if the waypoint path is intersecting with the shore polygon
            wp_line = LineString([(self.wp[each][1], self.wp[each][0]), (wp_y, wp_x)])
            if wp_line.intersects(enc.shore.geometry) == True:
                wp_x = self.wp[each][0]
                wp_y = self.wp[each][1]
            self.wp.append((wp_x, wp_y))

        return self.wp

    ###############################################
    # DYNAMICS
    ###############################################
    def update_states(self, dt):
        """
            Updates the states based on either a kinmatic model or dynamic model
        """
        if self.ship_model.use_kinematic_model:
            # updates the state
            self.kinematic_state_update(dt)
        else:
            self.eulersMethod(dt)
        
        # future position of the ship (used for visualization, checking for reaching the last waypoint and anti grounding)
        self.future_pos(10)

        # check for waypoints, grounding and update the los_angle
        self.update_reference()
    
    def kinematic_state_update(self, dt):
        """
            Updates the pose and turn rate based on a constrained kinematic model
            Heading == Course assumed
        """
        self.x += self.u * math.cos(self.psi) * dt
        self.y += self.u * math.sin(self.psi) * dt
        self.psi += self.r * dt
        self.psi = wrap_to_pi(self.psi)

        a = np.sign(self.U_d - self.u)*min(abs(self.U_d - self.u), self.ship_model.a_max)
        self.u += a*dt
        self.u = np.sign(self.u)*min(abs(self.u), self.ship_model.U_max)

        delta_psi = math.atan2(np.sin(self.los_angle - self.psi), np.cos(self.los_angle - self.psi))
        self.r = np.sign(delta_psi) * min(abs(delta_psi), self.ship_model.r_max)
    
    def future_pos(self, time):
        # Future position in defined time will be used to visualize ship's heading
        self.x_t = self.x + self.u * math.cos(self.psi) * time
        self.y_t = self.y + self.u * math.sin(self.psi) * time

    def eulersMethod(self, dt):
        """
        Dynamic ship model using ship parameters.
        :return: Returns ship states for each time step
        """
        # First of all update reference to find reference course and speed
        #self.update_reference()

        # rotation matrix elements
        r11, r12, r21, r22 = 0.0, 0.0, 0.0, 0.0

        self.psi = normalize_angle(self.psi)
        #psi_d = normalize_angle_diff(self.los_angle, self.psi)

        r11 = math.cos(self.psi)
        r12 = -math.sin(self.psi)
        r21 = math.sin(self.psi)
        r22 = math.cos(self.psi)

        self.ship_model.Cvv[0] = (-self.ship_model.M * self.v + self.ship_model.Y_vdot * self.v + self.ship_model.Y_rdot * self.r) * self.r
        self.ship_model.Cvv[1] = (self.ship_model.M * self.u - self.ship_model.X_udot * self.u) * self.r
        self.ship_model.Cvv[2] = ((self.ship_model.M * self.v - self.ship_model.Y_vdot * self.v - self.ship_model.Y_rdot * self.r) * self.u +
                                  (-self.ship_model.M * self.u + self.ship_model.X_udot * self.u) * self.v)

        self.ship_model.Dvv[0] = -(self.ship_model.X_u + self.ship_model.X_uu * math.fabs(self.u) + self.ship_model.X_uuu * self.u * self.u) * self.u
        self.ship_model.Dvv[1] = -((self.ship_model.Y_v * self.v + self.ship_model.Y_r * self.r) +
                               (self.ship_model.Y_vv * math.fabs(self.v) * self.v + self.ship_model.Y_vvv * self.v * self.v))
        self.ship_model.Dvv[2] = -((self.ship_model.N_v * self.v + self.ship_model.N_r * self.r) +
                               (self.ship_model.N_rr * math.fabs(self.r) * self.r + self.ship_model.N_rrr * self.r * self.r * self.r))

        self.updateCtrlInput() # updates tau

        self.x = self.x + dt * (r11 * self.u + r12 * self.v)
        self.y = self.y + dt * (r21 * self.u + r22 * self.v)
        self.psi = self.psi + dt * self.r
        self.psi = normalize_angle(self.psi)

        mu_dot = self.ship_model.Minv @ (self.ship_model.tau - self.ship_model.Cvv - self.ship_model.Dvv)
        mu_dot = mu_dot.flatten()

        self.u = self.u + dt * mu_dot[0]
        self.v = self.v + dt * mu_dot[1]
        self.r = self.r + dt * mu_dot[2]

    def updateCtrlInput(self):
        Fx = self.ship_model.Cvv[0] + self.ship_model.Dvv[0] + self.ship_model.Kp_u * self.ship_model.M * (self.U_d - self.u)
        delta_psi = math.atan2(np.sin(self.los_angle - self.psi), np.cos(self.los_angle - self.psi))
        Fy = ((self.ship_model.Kp_psi * self.ship_model.I_z) * (delta_psi - self.ship_model.Kd_psi * self.r)) / self.ship_model.rudder_dist

        # saturating controllers
        if Fx < self.ship_model.Fx_min:
            Fx = self.ship_model.Fx_min
        elif Fx > self.ship_model.Fx_max:
            Fx = self.ship_model.Fx_max

        if Fy < self.ship_model.Fy_min:
            Fy = self.ship_model.Fy_min
        elif Fy > self.ship_model.Fy_max:
            Fy = self.ship_model.Fy_max

        self.ship_model.tau[0] = Fx
        self.ship_model.tau[1] = Fy
        self.ship_model.tau[2] = self.ship_model.rudder_dist * Fy

    
    
    ###############################################
    # GUIDANCE/CONTROL
    ###############################################

    def update_reference(self):
        # check if the ship has reached the next waypoint (within radius of acceptance)
        if (self.wp[self.idx_next_wp][0]-self.x)**2 + (self.wp[self.idx_next_wp][1]-self.y)**2 <= self.R_a**2:
            self.idx_next_wp = min(self.idx_next_wp+1, len(self.wp)-1)
        
        self.update_los_angle()

        # Stop the ship if the future position is in the shore
        ship_pos = Point(self.y_t, self.x_t)
        if enc.shore.geometry.contains(ship_pos) == True:
            self.U_d = 0
        # check if the ship's position is 50 meters range of the last waypoint. If so, stop the ship.
        elif np.sqrt((self.wp[-1][0]-self.x)**2 + (self.wp[-1][1]-self.y)**2) <= 50:
            self.U_d = 0

    def update_los_angle(self):
        """
            Set the desired course based on Proportional LOS guidance law
        """
        # path-tangential angle
        pi_p = math.atan2(self.wp[self.idx_next_wp][1] - self.wp[self.idx_next_wp-1][1],
            self.wp[self.idx_next_wp][0] - self.wp[self.idx_next_wp-1][0])
        # cross-track error
        e = -np.sin(pi_p)*(self.x - self.wp[self.idx_next_wp-1][0])  + np.cos(pi_p)*(self.y - self.wp[self.idx_next_wp-1][1])
        c_d = pi_p + math.atan(-e/self.delta)
        c_d = wrap_to_pi(c_d)

        self.los_angle = c_d


    ###############################################
    # SITUATIONAL AWARENESS
    ###############################################

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
            x_est[2] = self.target_ship_state_est[i][2]*np.cos(self.target_ship_state_est[i][3])
            x_est[3] = self.target_ship_state_est[i][2]*np.sin(self.target_ship_state_est[i][3])
            x_est_list.append(x_est)
        return x_est_list


    
    
