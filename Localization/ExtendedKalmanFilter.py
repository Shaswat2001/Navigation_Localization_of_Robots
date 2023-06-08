import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))


import numpy as np
from MotionModel.simple_car.simple_car import SimpleCarModel
from ObservationModel.GPS.GPS import GPS

class EKF:

    def __init__(self,motionModel,obsModel):

        self.g = motionModel
        self.h = obsModel
        self.SIM_TIME = 50
        self.current_time = 0
        
        self.set_initial_conditions()

    def set_initial_conditions(self,X = None,P = None):

        if X:
            assert type(X) == np.ndarray, "State vector is not a matrix"
            assert len(X.shape) == 2, "State vector is not a 2D matrix"
            assert X.shape[0] == self.g.dim_state, "State vector is not of same size as that of dimensions of SIMPLE CAR MODEL"

            self.X = X
            self.P = P
        else:
            self.X = np.zeros((self.g.dim_state,1))
            self.P = np.eye(self.g.dim_state)

    def run(self):

        while self.current_time<self.SIM_TIME:

            pass

    def prediction_step(self):
        
        X = self.X
        P = self.P
        R = self.g.R

        X_prior,G = self.g.solve(X)
        P_prior = G.dot(P).dot(G.T) + R

        Z,h_x,H = self.h.solve(X_prior)

        return X_prior,P_prior
    
    def updatation_step(self,Xp,Pp):

        Q = self.h.Q
        Z,h_x,H = self.h.solve(Xp)

        K = self.calculate_kalman_gain(Pp,H,Q)

        X_post = X + K.dot(Z - h_x)
        F = np.eye(K.shape[0],H.shape[1]) - K@H
        P_post = F.dot(Pp)

        return X_post,P_post
    
    def calculate_kalman_gain(self,Pest,H,Q):
        
        y = H.dot(Pest).dot(H.T) + Q
        K = Pest.dot(H.T).dot(y)

        return K

    def reset(self,X = None,P = None):

        self.set_initial_conditions(X)



