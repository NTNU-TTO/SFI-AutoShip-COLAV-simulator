import math
import random
import numpy as np


def simulate_measurement(x_true: np.ndarray, cov: np.ndarray):
    """
        Input:
            x_true: [x_true, y_true, psi_true, u_true].T
            cov: WGN covariance matrix
    """
    return x_true + np.random.multivariate_normal(mean=x_true*0, cov=cov)


class Radar:
    """
    output:
        [x_meas, y_meas, psi_meas, u_meas].T
    """
    def __init__(self, meas_rate: float, accuracy: float, noise_prob: float, cov: np.ndarray):
        self.meas_rate = meas_rate
        self.accuracy = accuracy
        self.noise_prob = noise_prob
        self.cov = cov

    def simulate_measurement(self, x_true: np.ndarray, t: int):
        # returns None if t between measurements
        if t % self.meas_rate:
            return None # No AIS data recieved
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
    """

    def __init__(self, ais: AIS, radar: Radar):
        self.AIS = ais
        self.radar = radar

    def pred_step(self, x: np.ndarray, dt: float):
        """
            Simple straight line, constant speed prediction
            x: [x, y, psi, u].T
        """
        x[0] -= x[3] * math.sin(-x[2]) * dt
        x[1] += x[3] * math.cos(-x[2]) * dt
        return x

    def upd_step(self, x: np.ndarray, z: np.ndarray):
        """
            Simple mean of pred and measurement(s), given no Kalman Filter implementation
        """
        x_upd = x*0
        x_upd[0:2] = (x[0:2] + z[0:2])*0.5
        x_upd[2] = math.atan2((np.sin(x[2])+np.sin(z[2]))*0.5, (np.cos(x[2])+np.cos(z[2]))*0.5)
        x_upd[3] = (x[3] + z[3])*0.5
        return x_upd


    def step(self, x_true: np.ndarray, x_est: np.ndarray, t: int, dt: int):
        """
            Estimates the state of another ship for the next timestep
            x_true: true state
            x_est: last timestep estimate of state
        """
        x_prd = self.pred_step(x_est, dt)
        x_upd = x_prd
        z_ais = self.AIS.simulate_measurement(x_true, t)
        if z_ais is not None:
            x_prd = self.upd_step(x_upd, z_ais)
        z_radar = self.radar.simulate_measurement(x_true, t)
        if z_radar is not None:
            x_prd = self.upd_step(x_upd, z_radar)
        
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

ais = AIS(1, 2, 0.01, 0.1*np.eye(4))
radar = Radar(1, 2, 0.01, 0.1*np.eye(4))
x0 = np.array([50,50,0.1,5])
dt = 1

est = Estimator(ais, radar)
x_true = est.pred_step(x0, dt)
x = simulate_measurement(x0, np.diag([5,5,0.01,2]))
t = 2
x_est = est.step(x_true, x, t, dt)
print("x_true: ", x_true)
print("x_est: ", x_est)