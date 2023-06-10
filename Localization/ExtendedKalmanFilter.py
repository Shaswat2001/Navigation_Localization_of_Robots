import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
from MotionModel.SimpleCar import SimpleCarModel
from ObservationModel.GPS import GPS
from ObservationModel.RangeBearing import RangeBearing
from utils.utils import *

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
            self.Xt = X
            self.P = P
        else:
            self.X = np.zeros((self.g.dim_state,1))
            self.Xt = np.zeros((self.g.dim_state,1))
            self.P = np.eye(self.g.dim_state)

        self.results = {"true_path":self.Xt,
                        "est_path":self.X,
                        "observation":np.zeros((self.h.dim_z, 1))
                        }

    def run(self):

        while self.current_time<self.SIM_TIME:

            X_true,X_prior,P_prior = self.prediction_step()
            X_post,P_post,Z = self.updatation_step(X_prior,P_prior)

            self.X = X_post
            self.Xt = X_true
            self.P = P_post

            self.results["true_path"] = np.hstack((self.results["true_path"],X_true))
            self.results["est_path"] = np.hstack((self.results["est_path"],X_post))
            self.results["observation"] = np.hstack((self.results["observation"],Z))

            plot_paths(self.results,X_post,P_post)

            self.current_time+=self.g.DT

    def prediction_step(self):
        
        X = self.X
        X_true = self.Xt
        P = self.P

        X_true,_,_ = self.g.solve(X_true,"true")
        X_prior,G,R = self.g.solve(X,"true")
        P_prior = G @ P @ G.T + R

        return X_true,X_prior,P_prior
    
    def updatation_step(self,Xp,Pp):

        Q = self.h.Q
        Z,h_x,H = self.h.solve(Xp)

        K = self.calculate_kalman_gain(Pp,H,Q)

        res = residual(Z,h_x)
        X_post = Xp + K @ res
        F = np.eye(len(X_post)) - K @ H
        P_post = F @ Pp

        return X_post,P_post,Z
    
    def calculate_kalman_gain(self,Pest,H,Q):
        
        k_inv = np.linalg.inv(H @ Pest @ H.T + Q)
        K = Pest @ H.T @ k_inv

        return K

    def reset(self,X = None,P = None):

        self.set_initial_conditions(X,P)

if __name__ == "__main__":

    motion_model = SimpleCarModel()
    obs_model = GPS()
    
    obs_model = RangeBearing()

    landmarks = np.array([[5, 10], [10, 5], [15, 15]])
    filter = EKF(motion_model,obs_model)
    filter.run()

