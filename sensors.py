import math
import random
import numpy as np


def simulate_measurement(x_true: np.ndarray, cov: np.ndarray):
    """
        Input:
            x_true: [x_true, y_true, SOG_true, COG_true].T
            cov: WGN covariance matrix
    """
    return x_true + np.random.multivariate_normal(mean=x_true*0, cov=cov)


class Radar:
    """
    output:
        [x_meas, y_meas, SOG_meas, COG_meas].T
    """
    def __init__(self, meas_rate: float, accuracy: float, noise_prob: float, cov: np.ndarray):
        self.meas_rate = meas_rate
        self.accuracy = accuracy
        self.noise_prob = noise_prob
        self.cov = cov

    def simulate_measurement(self, x_true: np.ndarray, t: int):
        # returns None if t between measurements
        if t % self.meas_rate:
            return None
        else:
            return simulate_measurement(x_true, self.cov)
        
class AIS:

    def __init__(self, meas_rate: float, accuracy: float, loss_prob: float, cov: np.ndarray):
        self.meas_rate = meas_rate
        self.accuracy = accuracy
        self.loss_prob = loss_prob
        self.cov = cov

    def simulate_measurement(self, x_true: np.ndarray, t: int):
        # returns None if no meas recieved/ t between measurements
        if self.loss_prob > random.uniform(0,1) or t % self.meas_rate:
            return None
        else:
            return simulate_measurement(x_true, self.cov)

class Estimator:
    """
        Estimates the states of the other ships
        No Kalman Filter used
        x: [x, y, SOG, COG]
    """

    def __init__(self, sensor_list):
        #Takes in list of sensor objects, ex: sensor_list = [ais: AIS, radar: Radar]
        self.sensors = sensor_list

    def pred_step(self, x: np.ndarray, dt: float):
        """
            Simple straight line, constant speed prediction
            x: [x, y, psi, u].T
        """
        x[0] += x[3] * math.cos(x[2]) * dt
        x[1] += x[3] * math.sin(x[2]) * dt
        return x

    def upd_step(self, x: np.ndarray, z: np.ndarray):
        """
            Simple mean of pred and measurement(s), given no Kalman Filter implementation
        """
        x_upd = x*0
        x_upd[0:3] = (x[0:3] + z[0:3])*0.5
        x_upd[3] = math.fmod(math.atan2(np.sin(x[3])+np.sin(z[3]), np.cos(x[3])+np.cos(z[3])) + 2 * math.pi, 2 * math.pi)
        return x_upd


    def step(self, x_true: np.ndarray, x_est: np.ndarray, t: int, dt: int):
        """
            Estimates the state of another ship for the next timestep
            x_true: true state
            x_est: last timestep estimate of state

            x_true could possibly be replaced with z, removing the simulation of measurements from the estimator
        """
        x_prd = self.pred_step(x_est, dt)
        x_upd = x_prd
        for sensor in self.sensors:
            z = sensor.simulate_measurement(x_true, t)

            if z is not None:
                x_upd = self.upd_step(x_upd, z)
        return x_upd





"""
    Radar:
    rate; accuracy; noise probabilities, predictor

    AIS/VDES:
    rate; accuracy; loss probabilities; predictor

        rate:
        Class A	Anchored / Moored	 Every 3 Minutes
        Class A	Sailing 0-14 knots	 Every 10 Seconds
        Class A	Sailing 14-23 knots	 Every 6 Seconds
        Class A	Sailing 0-14 knots and changing course	 Every 3.33 Seconds
        Class A	Sailing 14-23 knots and changing course	 Every 2 Seconds
        Class A	Sailing faster than 23 knots	 Every 2 Seconds
        Class A	Sailing faster than 23 knots and changing course	 Every 2 Seconds
        Class B	Stopped or sailing up to 2 knots	 Every 3 Minutes
        Class B	Sailing faster than 2 knots	 Every 30 Seconds

    Camera?
"""
