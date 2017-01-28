from math import *
import numpy as np

class KalmanFilter:
    def __init__(self, u = np.array([[0.], [0.]]),  # external motion 
                 F = np.array([[1., 1.], [0, 1.]]), # next state function
                 H = np.array([[1., 0.]]),          # measurement function
                 R = np.array([[1.]]),              # measurement uncertainty
                 Q = 0.,
                 G = 0.):
        self.u = u
        self.F = F
        self.H = H
        self.R = R
        self.G = G
        self.Q = Q

    def step(self, x, P, measurement):
        I = np.array([[1., 0.], [0., 1.]])
        z = np.array([[measurement]])

        # Innovation (output error): y = z - H*x
        # Output Error covariance: S = H*P*H' + R
        # Kalman gain: K = P*H'*inv(S)
        # Correction of state estimation: x = x + K*y
        # Correction of estimation of error covariance: P = P - K*H*P
        # State prediction: x = F*x
        # Error covariance prediction: P = F*P*F' + G*Q*G'

        # measurement update
        y = z - np.dot(self.H, x)
        S = np.dot(self.H, np.dot(P, self.H.transpose())) + self.R
        K = np.dot(P, self.H.transpose()) * np.linalg.inv(S)
        x = x + np.dot(K, y)

        P = np.dot((I-np.dot(K, self.H)), P)
        
        # prediction
        x = np.dot(self.F, x) + self.u
        P = np.dot(self.F, np.dot(P, self.F.transpose())) + np.dot(self.G, np.dot(self.Q, self.G.transpose()))
        return x, P
