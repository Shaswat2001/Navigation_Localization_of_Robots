import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
from MotionModel.SimpleCar import SimpleCarModel
from ObservationModel.GPS import GPS
from utils.utils import *

class UKF:

    def __init__(self,motionModel,obsModel):

        self.g = motionModel
        self.h = obsModel
        self.SIM_TIME = 50
        self.current_time = 0

        self.alpha = 0.3
        self.n = motionModel.dim_state
        self.k = 0.1
        self.beta = 2

        self.set_initial_conditions()

    def set_initial_conditions(self,X = None,P = None):

        if X:
            assert type(X) == np.ndarray, "State vector is not a matrix"
            assert len(X.shape) == 2, "State vector is not a 2D matrix"
            assert X.shape[0] == self.g.dim_state, "State vector is not of same size as that of dimensions of SIMPLE CAR MODEL"

            Xinit = X
            Xinit_true = X
            Pinit = P

        else:
            Pinit = np.eye(self.g.dim_state)
            Xinit = np.zeros((self.g.dim_state,1))
            Xinit_true = np.zeros((self.g.dim_state,1))


        self.P = Pinit
        self.X = self.get_sigma(Xinit,self.P)
        self.Xt = self.get_sigma(Xinit_true,self.P)
        
        self.calculate_weights()

        self.results = {"true_path":self.Xt,
                        "est_path":self.X,
                        "observation":np.zeros((self.h.dim_z, 1))
                        }
    
    def calculate_weights(self):

        lmbd = (self.n +  self.k)*(self.alpha**2) - self.n

        self.w_m = [lmbd/(self.n + lmbd)]
        self.w_c = [lmbd/(self.n + lmbd) + (1 - self.alpha**2 + self.beta)]

        for i in range(2*self.n):

            self.w_c.append(1/(2*(self.n + lmbd)))
            self.w_m.append(1/(2*(self.n + lmbd)))

    def run(self):

        while self.current_time<self.SIM_TIME:

            X_true,X_prior,P_prior = self.prediction_step()
            X_prior_sigma = self.get_sigma(X_prior,P_prior)
            X_post,P_post,Z = self.updatation_step(X_prior_sigma,X_prior,P_prior)

            self.results["true_path"] = np.hstack((self.results["true_path"],X_true))
            self.results["est_path"] = np.hstack((self.results["est_path"],X_post))
            self.results["observation"] = np.hstack((self.results["observation"],Z))

            # plot_paths(self.results,X_post,P_post)

            self.X = self.get_sigma(X_post,P_post)
            self.Xt = self.get_sigma(X_true,P_post)
            self.P = P_post

            self.current_time+=self.g.DT

    def prediction_step(self):
        
        X = self.X
        X_true = self.Xt
        P = self.P
        R = self.g.R

        X_true_sigma,_ = self.g.solve(X_true.T)
        X_prior_sigma,_ = self.g.solve(X.T)

        print(X_prior_sigma.shape)
        X_true = self.calculate_weight_sum(X_true_sigma)
        X_prior = self.calculate_weight_sum(X_prior_sigma)

        P_sigma = self.calculate_estimated_variance(X_prior_sigma,X_prior)
        P_prior = P_sigma + R

        return X_true,X_prior,P_prior
    
    def updatation_step(self,Xp,Xp_sum,Pp):

        Q = self.h.Q
        Z,h_x_sigma,H = self.h.solve(Xp.T)

        h_x = self.calculate_weight_sum(h_x_sigma)

        K,S= self.calculate_kalman_gain(Xp,Xp_sum,h_x_sigma,h_x,Q)

        residual = Z - h_x
        X_post = Xp + K @ residual
        P_post = Pp - K @ S @ K.T

        return X_post,P_post,Z
    
    def calculate_kalman_gain(self,Xp,Xp_sum,h_x_sigma,h_x,Q):
        
        St = Q
        Vxz = np.zeroes(Xp.shape[0],h_x.shape[0])

        for i in range(len(self.w_c)):

            resS = h_x_sigma[i].reshape(h_x.shape[0],1) - h_x
            resX = Xp_sum[i].reshape(Xp.shape[0],1) - Xp

            St += self.w_c[i]*resS @ resS.T
            Vxz += self.w_c[i]*resX @ resS.T

        K = Vxz @ np.linalg.inv(St)

        return K,St

    def reset(self,X = None,P = None):

        self.set_initial_conditions(X,P)
    
    def get_sigma(self,X,P):

        sigma_points = np.zeros((2*self.n+1,X.shape[0]))

        lmbd = (self.n +  self.k)*(self.alpha**2) - self.n

        U = np.sqrt((self.n + lmbd)*P)
        sigma_points[0] = X[:,0]
        for i in range(self.n):
            arr = U[:,i].reshape(U.shape[0],1)
            sigma_points[i+1] = (X + arr)[:,0]
            sigma_points[i+1+self.n] = (X - arr)[:,0]

        return sigma_points
    
    def calculate_weight_sum(self,points):

        sum_pt = np.zeros((points.shape[1],1))

        for i in range(len(self.w_m)):

            sum_pt += self.w_m[i]*points[i,:].reshape(points.shape[1],1)

        return sum_pt
    
    def calculate_estimated_variance(self,Xs,X):

        variance = np.zeros((X.shape[0],X.shape[0]))

        for i in range(len(self.w_c)):

            res = Xs[i].reshape(X.shape[0],1) - X

            variance += self.w_c[i]*res @ res.T

        return variance

if __name__ == "__main__":

    motion_model = SimpleCarModel()
    obs_model = GPS()
    filter = UKF(motion_model,obs_model)
    filter.run()

