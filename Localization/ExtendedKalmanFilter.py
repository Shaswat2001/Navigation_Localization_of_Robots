import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))


import numpy as np
from MotionModel.simple_car.simple_car import SimpleCarModel
from ObservationModel.GPS.GPS import GPS
from utils.utils import *
import matplotlib.pyplot as plt

class EKF:

    def __init__(self,motionModel,obsModel):

        self.g = motionModel
        self.h = obsModel
        self.SIM_TIME = 50
        self.current_time = 0

        self.Xtrue_list = []
        self.Xest_list = []
        self.obs_list = []
        
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
            self.Xtrue = np.zeros((self.g.dim_state,1))
            self.P = np.eye(self.g.dim_state)

    def run(self):

        while self.current_time<self.SIM_TIME:
            

            Xt,Xp,Pp = self.prediction_step()
            X_post,P_post,Z = self.updatation_step(Xp,Pp)

            self.X = X_post
            self.Xtrue = Xt
            self.P = P_post

            self.Xest_list.append(X_post)
            self.Xtrue_list.append(Xt)

            plot_paths(self.Xtrue_list,self.Xest_list,self.obs_list,X_post,P_post)

            self.current_time+=self.g.DT

    def prediction_step(self):
        
        X = self.X
        Xtrue = self.Xtrue
        P = self.P
        R = self.g.R

        Xtrue,_ = self.g.solve(Xtrue)
        X_prior,G = self.g.solve(X)
        P_prior = G.dot(P).dot(G.T) + R

        return Xtrue,X_prior,P_prior
    
    def updatation_step(self,Xp,Pp):

        Q = self.h.Q
        Z,h_x,H = self.h.solve(Xp)

        K = self.calculate_kalman_gain(Pp,H,Q)

        X_post = Xp + K.dot(Z - h_x)
        F = np.eye(K.shape[0],H.shape[1]) - K.dot(H)
        P_post = F.dot(Pp)

        self.obs_list.append(Z)

        return X_post,P_post,Z
    
    def calculate_kalman_gain(self,Pest,H,Q):
        
        y = np.linalg.inv(H.dot(Pest).dot(H.T) + Q)
        K = Pest.dot(H.T).dot(y)

        return K

    def reset(self,X = None,P = None):

        self.set_initial_conditions(X,P)

if __name__ == "__main__":

    motion_model = SimpleCarModel()
    obs_model = GPS()
    filter = EKF(motion_model,obs_model)
    filter.run()
    # plot_paths(filter.Xtrue_list,filter.Xest_list,filter.obs_list)


