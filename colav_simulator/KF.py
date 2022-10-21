import math
import random
import numpy as np
import scipy.linalg as la

# x = [x, y, Vx, Vy]

### Dynamic model(s) ###
class CV_model:

    def __init__(self, sigma_a: float = 0.5):
        self.sigma_a = sigma_a

    def F(self, Ts: float):
        F = np.eye(4)
        F[:2, 2:] = Ts*np.eye(2)
        return F
    def Q(self, Ts):
        A = (1/3)*Ts**3 * np.eye(2)
        B = (1/2)*Ts**2 * np.eye(2)
        D = Ts * np.eye(2)
        Q = np.block([[A, B],
                    B, D])*self.sigma_a**2
        return Q
        
    def f(self, x: np.ndarray, Ts: float):
        return self.F(Ts) @ x

### Measurement model(s) ###
class Radar:
    """
    output:
        [x_meas, y_meas].T ([x_meas, y_meas, Vx_meas, Vy_meas].T if meas_dim=4)
    """
    def __init__(self, meas_rate: float, sigma_z: float, meas_dim = 2):
        self.meas_rate = meas_rate
        self.sigma_z = sigma_z
        self.meas_dim = meas_dim # =2 for only pos, =4 for pos and vel
        self._R = self.sigma_z**2 * np.eye(self.meas_dim)
        if self.meas_dim == 2:
            self._H = np.block([np.eye(2), np.zeros((2,2))])
        else:
            self._H = np.eye(4)
    def R(self, x: np.ndarray):
        return self._R

    def H(self):
        return self._H

    def h(self, x: np.ndarray):
        return self._H @ x

    def simiulate_measurement(self, x_true: np.ndarray, t: float):
        if t % self.meas_rate:
            return None
        else:
            z = np.random.multivariate_normal(mean=self._H@x_true, cov=self._R)
        

class AIS:
    """
    output:
        [x_meas, y_meas, Vx_meas, Vy_meas].T
    """
    def __init__(self, meas_rate: float, loss_prob: float, sigma_z: float):
        self.meas_rate = meas_rate
        self.loss_prob = loss_prob
        self.sigma_z = sigma_z
        self._R = self.sigma_z**2 * np.eye(4)
        self._H = np.eye(4)

    # R_realistic values from paper considering quantization effects
    # https://link.springer.com/chapter/10.1007/978-3-319-55372-6_13#Sec14
    def R_realistic(self, x: np.ndarray):
        R_GNSS = np.diag([0.5, 0.5, 0.1, 0.1])**2
        R_v = np.diag([x[2]**2, x[3]**2, 0, 0])
        self._R = R_GNSS + (1/12)*R_v

    def R(self, x: np.ndarray):
        return self._R

    def H(self):
        return self._H

    def h(self, x: np.ndarray):
        z = self._H @ x
        return z

    def simiulate_measurement(self, x_true: np.ndarray, t: float):
        if t % self.meas_rate or random.uniform(0,1) < self.loss_prob:
            return None
        else:
            z = np.random.multivariate_normal(mean=self._H@x_true, cov=self.R(x_true))
    
### Kalman Filter ###
class KF:
    """
        Estimates state of one ship
        prediction_model: CV model
        sensor_model = Radar or AIS
    """
    def __init__(self, prediction_model, sensor_model):
        self.pred_model = prediction_model
        self.sensor_model = sensor_model

    def predict(self, x: np.ndarray, P: np.ndarray, Ts: float):
        F = self.pred_model.F(Ts)
        Q = self.pred_model.Q(Ts)
        f = self.pred_model.f(x, Ts)

        x_pred = self.pred_model.f(x, Ts)
        P_pred = F @ P @ F.T + Q

        return x_pred, P_pred

    def innovation(self, x: np.ndarray, P: np.ndarray, z: np.ndarray):
        zbar = self.sensor_model.h(x)  # predicted measurement
        v = z - zbar # innovation mean

        H = self.sensor_model.H()
        R = self.sensor_model.R(x)
        S = H @ P @ H.T + R  # innovation covariance

        return v, S

    def update(self, x: np.ndarray, P: np.ndarray, z: np.ndarray):
        v, S = self.innovation(x, P, z)
        H = self.sensor_model.H()

        W = P @ la.solve(S, H).T
        x_upd = x + W @ v
        P_upd = P - W @ H @ P

        return x_upd, P_upd

    def step(self, x: np.ndarray, P: np.ndarray, z: np.ndarray, Ts: float):
        x_pred, P_pred = self.predict(x, P, Ts)
        x_upd, P_upd = self.update(x_pred, P_pred, z)
        return x_upd, P_upd
